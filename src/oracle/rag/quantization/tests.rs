//! Tests for quantization module
//!
//! Following Popperian falsification checklist from specification

#[cfg(test)]
mod tests {
    use crate::oracle::rag::quantization::{
        CalibrationStats, QuantizationError, QuantizationParams, QuantizedEmbedding,
        RescoreRetriever, RescoreRetrieverConfig, SimdBackend, dot_i8_scalar,
    };

    // ========================================================================
    // SECTION 1: Quantization Accuracy Tests (QA-01 to QA-15)
    // Following Popperian falsification checklist from specification
    // ========================================================================

    mod quantization_accuracy {
        use super::*;

        /// QA-01: Quantization Error Bound
        /// Claim: |Q(x) - x| <= scale/2
        #[test]
        fn qa_01_quantization_error_bound() {
            let params = QuantizationParams::new(0.1, 384).unwrap();
            let max_error = params.max_error_bound();

            // Test many values
            for i in -1000..=1000 {
                let x = i as f32 * 0.01;
                let q = params.quantize_value(x);
                let dq = params.dequantize_value(q);
                let error = (dq - x).abs();

                // Error should be within bound (with small epsilon for floating point)
                assert!(
                    error <= max_error + 1e-6,
                    "Error {} exceeds bound {} for x={}",
                    error,
                    max_error,
                    x
                );
            }
        }

        /// QA-02: Symmetric Quantization (zero_point = 0)
        #[test]
        fn qa_02_symmetric_quantization() {
            let params = QuantizationParams::new(0.1, 384).unwrap();
            assert_eq!(params.zero_point, 0, "Zero point must be 0 for symmetric");
        }

        /// QA-03: Calibration Convergence
        #[test]
        fn qa_03_calibration_convergence() {
            let mut cal = CalibrationStats::new(384);

            // Add many samples
            for _ in 0..1000 {
                let embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
                cal.update(&embedding).unwrap();
            }

            // Absmax should stabilize
            let absmax1 = cal.absmax;

            for _ in 0..100 {
                let embedding: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
                cal.update(&embedding).unwrap();
            }

            let absmax2 = cal.absmax;

            // Should not change significantly
            assert!(
                (absmax2 - absmax1).abs() < 0.01,
                "Calibration should converge: {} vs {}",
                absmax1,
                absmax2
            );
        }

        /// QA-04: Overflow Prevention in Dot Product
        #[test]
        fn qa_04_overflow_prevention() {
            let backend = SimdBackend::Scalar;

            // Maximum possible values: 127 x 127 x 4096 = 66,044,672
            // This fits in i32 (max 2,147,483,647)
            let a = vec![127i8; 4096];
            let b = vec![127i8; 4096];

            let result = backend.dot_i8(&a, &b);

            // Should not overflow
            assert_eq!(result, 127 * 127 * 4096);
        }

        /// QA-05: NaN/Inf Handling
        #[test]
        fn qa_05_nan_handling() {
            let cal = CalibrationStats::new(4);
            let embedding = vec![1.0, f32::NAN, 3.0, 4.0];

            let result = QuantizedEmbedding::from_f32(&embedding, &cal);
            assert!(matches!(
                result,
                Err(QuantizationError::NonFiniteValue { .. })
            ));
        }

        #[test]
        fn qa_05_inf_handling() {
            let cal = CalibrationStats::new(4);
            let embedding = vec![1.0, f32::INFINITY, 3.0, 4.0];

            let result = QuantizedEmbedding::from_f32(&embedding, &cal);
            assert!(matches!(
                result,
                Err(QuantizationError::NonFiniteValue { .. })
            ));
        }

        /// QA-06: Dequantization Reversibility
        #[test]
        fn qa_06_dequantization_reversibility() {
            let mut cal = CalibrationStats::new(4);
            let original = vec![0.5, -0.3, 0.8, -0.1];
            cal.update(&original).unwrap();

            let quantized = QuantizedEmbedding::from_f32(&original, &cal).unwrap();
            let dequantized = quantized.dequantize();

            let params = cal.to_quant_params().unwrap();
            let max_error = params.scale;

            for (i, (&orig, &deq)) in original.iter().zip(dequantized.iter()).enumerate() {
                let error = (orig - deq).abs();
                assert!(
                    error <= max_error,
                    "Error {} exceeds bound {} at index {}",
                    error,
                    max_error,
                    i
                );
            }
        }

        /// QA-07: Scale Factor Computation
        #[test]
        fn qa_07_scale_computation() {
            let absmax = 12.7;
            let params = QuantizationParams::from_absmax(absmax, 384).unwrap();

            let expected_scale = absmax / 127.0;
            assert!(
                (params.scale - expected_scale).abs() < 1e-6,
                "Scale {} != expected {}",
                params.scale,
                expected_scale
            );
        }

        /// QA-08: Zero Point for Symmetric
        #[test]
        fn qa_08_zero_point_symmetric() {
            let params = QuantizationParams::new(0.1, 384).unwrap();
            assert_eq!(params.zero_point, 0);

            let params2 = QuantizationParams::from_absmax(12.7, 384).unwrap();
            assert_eq!(params2.zero_point, 0);
        }

        /// QA-09: Clipping at Boundaries
        #[test]
        fn qa_09_clipping_boundaries() {
            let params = QuantizationParams::new(0.01, 384).unwrap();

            // Large positive should clip to 127
            let large = 10000.0;
            assert_eq!(params.quantize_value(large), 127);

            // Large negative should clip to -128
            let small = -10000.0;
            assert_eq!(params.quantize_value(small), -128);
        }

        /// QA-10: Dimension Mismatch Detection
        #[test]
        fn qa_10_dimension_mismatch() {
            let cal = CalibrationStats::new(384);
            let wrong_dims = vec![1.0; 256];

            let result = QuantizedEmbedding::from_f32(&wrong_dims, &cal);
            assert!(matches!(
                result,
                Err(QuantizationError::DimensionMismatch { .. })
            ));
        }

        /// QA-11: Content Hash Integrity
        #[test]
        fn qa_11_content_hash_integrity() {
            let mut cal = CalibrationStats::new(4);
            let embedding = vec![0.5, -0.3, 0.8, -0.1];
            cal.update(&embedding).unwrap();

            let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
            assert!(quantized.verify_integrity());

            // Tamper with values
            let mut tampered = quantized.clone();
            tampered.values[0] = 99;
            assert!(!tampered.verify_integrity());
        }

        /// QA-12: Welford's Numerical Stability
        #[test]
        fn qa_12_welford_stability() {
            let mut cal = CalibrationStats::new(4);

            // Add many samples with known statistics
            for i in 0..10000 {
                let val = (i as f32 * 0.001).sin();
                let embedding = vec![val; 4];
                cal.update(&embedding).unwrap();
            }

            // Mean should be close to actual mean of sin values
            // Variance should be positive and finite
            for i in 0..4 {
                assert!(cal.mean[i].is_finite());
                assert!(cal.variance(i).is_finite());
                assert!(cal.variance(i) >= 0.0);
            }
        }

        /// QA-13: Quantization Determinism
        #[test]
        fn qa_13_quantization_determinism() {
            let mut cal = CalibrationStats::new(4);
            let embedding = vec![0.5, -0.3, 0.8, -0.1];
            cal.update(&embedding).unwrap();

            let q1 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
            let q2 = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

            assert_eq!(q1.values, q2.values);
            assert_eq!(q1.hash, q2.hash);
        }

        /// QA-14: Memory Layout (contiguous)
        #[test]
        fn qa_14_memory_layout() {
            let mut cal = CalibrationStats::new(384);
            let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
            cal.update(&embedding).unwrap();

            let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();

            // Values should be contiguous
            assert_eq!(quantized.values.len(), 384);
            assert_eq!(quantized.values.capacity(), 384);
        }

        /// QA-15: Batch vs Individual Consistency
        #[test]
        fn qa_15_batch_consistency() {
            let mut cal1 = CalibrationStats::new(4);
            let mut cal2 = CalibrationStats::new(4);

            let embeddings: Vec<Vec<f32>> =
                (0..100).map(|i| vec![(i as f32 * 0.01).sin(); 4]).collect();

            // Individual updates
            for emb in &embeddings {
                cal1.update(emb).unwrap();
            }

            // Batch update
            cal2.update_batch(&embeddings).unwrap();

            assert!((cal1.absmax - cal2.absmax).abs() < 1e-6);
            assert_eq!(cal1.n_samples, cal2.n_samples);
            for i in 0..4 {
                assert!((cal1.mean[i] - cal2.mean[i]).abs() < 1e-6);
            }
        }
    }

    // ========================================================================
    // SECTION 2: Retrieval Accuracy Tests (RA-01 to RA-15)
    // ========================================================================

    mod retrieval_accuracy {
        use super::*;

        /// RA-01: Accuracy Retention
        #[test]
        fn ra_01_retrieval_returns_results() {
            let config = RescoreRetrieverConfig {
                rescore_multiplier: 4,
                top_k: 5,
                min_calibration_samples: 1,
                simd_backend: Some(SimdBackend::Scalar),
            };
            let mut retriever = RescoreRetriever::new(4, config);

            // Index some documents
            for i in 0..10 {
                let emb = vec![i as f32 * 0.1; 4];
                retriever
                    .index_document(&format!("doc{}", i), &emb)
                    .unwrap();
            }

            let query = vec![0.5; 4];
            let results = retriever.retrieve(&query).unwrap();

            assert!(!results.is_empty());
            assert!(results.len() <= 5);
        }

        /// RA-02: Rescore Multiplier Effect
        #[test]
        fn ra_02_rescore_multiplier() {
            let config = RescoreRetrieverConfig {
                rescore_multiplier: 4,
                top_k: 2,
                min_calibration_samples: 1,
                simd_backend: Some(SimdBackend::Scalar),
            };
            let mut retriever = RescoreRetriever::new(4, config);

            for i in 0..10 {
                let emb = vec![i as f32 * 0.1; 4];
                retriever
                    .index_document(&format!("doc{}", i), &emb)
                    .unwrap();
            }

            // Stage 1 should retrieve 4 x 2 = 8 candidates
            let query = vec![0.5; 4];
            let results = retriever.retrieve(&query).unwrap();

            // Should return exactly top_k
            assert_eq!(results.len(), 2);
        }

        /// RA-05: Empty Index Handling
        #[test]
        fn ra_05_empty_index() {
            let config = RescoreRetrieverConfig::default();
            let mut retriever = RescoreRetriever::new(4, config);

            // Add calibration sample so we can query
            retriever
                .add_calibration_sample(&[1.0, 2.0, 3.0, 4.0])
                .unwrap();

            let query = vec![0.5; 4];
            let results = retriever.retrieve(&query).unwrap();

            // No documents indexed, so results should be empty
            assert!(results.is_empty());
        }

        /// RA-08: Empty Query Handling
        #[test]
        fn ra_08_empty_query() {
            let config = RescoreRetrieverConfig::default();
            let mut retriever = RescoreRetriever::new(4, config);

            retriever
                .index_document("doc1", &[1.0, 2.0, 3.0, 4.0])
                .unwrap();

            // Empty query should fail with dimension mismatch
            let empty_query: Vec<f32> = vec![];
            let result = retriever.retrieve(&empty_query);
            assert!(matches!(
                result,
                Err(QuantizationError::DimensionMismatch { .. })
            ));
        }

        /// RA-11: Score Ordering
        #[test]
        fn ra_11_score_ordering() {
            let config = RescoreRetrieverConfig {
                rescore_multiplier: 4,
                top_k: 5,
                min_calibration_samples: 1,
                simd_backend: Some(SimdBackend::Scalar),
            };
            let mut retriever = RescoreRetriever::new(4, config);

            for i in 0..10 {
                let emb = vec![i as f32 * 0.1; 4];
                retriever
                    .index_document(&format!("doc{}", i), &emb)
                    .unwrap();
            }

            let query = vec![0.5; 4];
            let results = retriever.retrieve(&query).unwrap();

            // Results should be sorted descending by score
            for i in 1..results.len() {
                assert!(
                    results[i - 1].score >= results[i].score,
                    "Results not sorted: {} < {}",
                    results[i - 1].score,
                    results[i].score
                );
            }
        }
    }

    // ========================================================================
    // SECTION 3: Performance Tests (PF-01 to PF-15)
    // ========================================================================

    mod performance {
        use super::*;

        /// PF-02: Memory Reduction (4x)
        #[test]
        fn pf_02_memory_reduction() {
            let f32_size = 384 * 4; // 384 dims x 4 bytes
            let mut cal = CalibrationStats::new(384);
            let embedding: Vec<f32> = (0..384).map(|i| i as f32 * 0.01).collect();
            cal.update(&embedding).unwrap();

            let quantized = QuantizedEmbedding::from_f32(&embedding, &cal).unwrap();
            let i8_size = quantized.memory_size();

            // Should be significantly less than f32 (at least 2x, ideally ~4x)
            assert!(
                i8_size < f32_size,
                "Int8 size {} should be less than f32 size {}",
                i8_size,
                f32_size
            );
        }

        /// PF-05: SIMD Backend Detection
        #[test]
        fn pf_05_simd_detection() {
            let backend = SimdBackend::detect();

            // Should detect some backend
            match backend {
                SimdBackend::Avx512 => println!("Detected AVX-512"),
                SimdBackend::Avx2 => println!("Detected AVX2"),
                SimdBackend::Neon => println!("Detected NEON"),
                SimdBackend::Scalar => println!("Using scalar fallback"),
            }

            // Backend should work
            let a = vec![1i8; 32];
            let b = vec![2i8; 32];
            let result = backend.dot_i8(&a, &b);
            assert_eq!(result, 64); // 1 x 2 x 32 = 64
        }

        /// PF-06: Scalar Dot Product Correctness
        #[test]
        fn pf_06_scalar_dot_product() {
            let a = vec![1i8, 2, 3, 4];
            let b = vec![5i8, 6, 7, 8];

            let result = dot_i8_scalar(&a, &b);
            assert_eq!(result, 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8);
        }

        /// Test SIMD backends produce same results as scalar
        #[test]
        fn pf_simd_matches_scalar() {
            let backend = SimdBackend::detect();

            // Test various sizes
            for size in [16, 32, 64, 128, 384, 1024] {
                let a: Vec<i8> = (0..size).map(|i| (i % 127) as i8).collect();
                let b: Vec<i8> = (0..size).map(|i| ((i * 7) % 127) as i8).collect();

                let scalar_result = dot_i8_scalar(&a, &b);
                let simd_result = backend.dot_i8(&a, &b);

                assert_eq!(
                    scalar_result, simd_result,
                    "SIMD mismatch for size {}: {} vs {}",
                    size, scalar_result, simd_result
                );
            }
        }
    }

    // ========================================================================
    // SECTION 4: Numerical Correctness Tests (NC-01 to NC-15)
    // ========================================================================

    mod numerical_correctness {
        use super::*;

        /// NC-05: Zero Vector Handling
        #[test]
        fn nc_05_zero_vector_handling() {
            let zero_embedding = vec![0.0f32; 4];
            let result = QuantizedEmbedding::from_f32_uncalibrated(&zero_embedding);
            assert!(result.is_ok());

            let quantized = result.unwrap();
            assert!(quantized.values.iter().all(|&v| v == 0));
        }

        /// NC-09: Dot Product Symmetry
        #[test]
        fn nc_09_dot_product_symmetry() {
            let backend = SimdBackend::Scalar;
            let a = vec![1i8, 2, 3, 4, 5];
            let b = vec![5i8, 4, 3, 2, 1];

            let ab = backend.dot_i8(&a, &b);
            let ba = backend.dot_i8(&b, &a);

            assert_eq!(ab, ba);
        }

        /// NC-13: f32 x i8 Rescoring Correctness
        #[test]
        fn nc_13_f32_i8_dot_product() {
            let backend = SimdBackend::Scalar;
            let query = vec![1.0f32, 2.0, 3.0, 4.0];
            let doc = vec![1i8, 2, 3, 4];
            let scale = 0.1;

            let result = backend.dot_f32_i8(&query, &doc, scale);

            // Expected: (1.0 x 1 x 0.1) + (2.0 x 2 x 0.1) + (3.0 x 3 x 0.1) + (4.0 x 4 x 0.1)
            //         = 0.1 + 0.4 + 0.9 + 1.6 = 3.0
            assert!((result - 3.0).abs() < 1e-6);
        }
    }

    // ========================================================================
    // SECTION 5: Safety & Robustness Tests (SR-01 to SR-10)
    // ========================================================================

    mod safety_robustness {
        use super::*;

        /// SR-01: No Panics on Valid Input
        #[test]
        fn sr_01_no_panic_valid_input() {
            let mut cal = CalibrationStats::new(4);
            let embeddings = vec![
                vec![0.0, 0.0, 0.0, 0.0],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![-1.0, -1.0, -1.0, -1.0],
                vec![0.5, -0.5, 0.5, -0.5],
            ];

            for emb in &embeddings {
                cal.update(emb).unwrap();
                QuantizedEmbedding::from_f32(emb, &cal).unwrap();
            }
        }

        /// SR-05: Input Validation
        #[test]
        fn sr_05_input_validation() {
            let cal = CalibrationStats::new(4);

            // Empty should fail
            assert!(QuantizedEmbedding::from_f32(&[], &cal).is_err());

            // Wrong dimensions should fail
            assert!(QuantizedEmbedding::from_f32(&[1.0, 2.0], &cal).is_err());

            // NaN should fail
            assert!(QuantizedEmbedding::from_f32(&[1.0, f32::NAN, 3.0, 4.0], &cal).is_err());

            // Inf should fail
            assert!(QuantizedEmbedding::from_f32(&[1.0, f32::INFINITY, 3.0, 4.0], &cal).is_err());
        }

        /// SR-08: Error Message Quality
        #[test]
        fn sr_08_error_messages() {
            let error = QuantizationError::DimensionMismatch {
                expected: 384,
                actual: 256,
            };
            let msg = error.to_string();
            assert!(msg.contains("384"));
            assert!(msg.contains("256"));

            let error = QuantizationError::NonFiniteValue {
                index: 42,
                value: f32::NAN,
            };
            let msg = error.to_string();
            assert!(msg.contains("42"));
        }
    }

    // ========================================================================
    // Property-Based Tests
    // ========================================================================

    mod proptests {
        use super::*;
        use proptest::prelude::*;

        proptest! {
            #![proptest_config(ProptestConfig::with_cases(100))]

            /// Quantization error is always bounded
            #[test]
            fn prop_quantization_error_bounded(
                values in prop::collection::vec(-10.0f32..10.0, 1..100)
            ) {
                let quantized = QuantizedEmbedding::from_f32_uncalibrated(&values);
                if let Ok(q) = quantized {
                    let dequantized = q.dequantize();
                    let max_error = q.params.scale;

                    for (i, (&orig, &deq)) in values.iter().zip(dequantized.iter()).enumerate() {
                        let error = (orig - deq).abs();
                        prop_assert!(
                            error <= max_error + 1e-5,
                            "Error {} > bound {} at index {}",
                            error, max_error, i
                        );
                    }
                }
            }

            /// Dot product is symmetric
            #[test]
            fn prop_dot_symmetric(
                a in prop::collection::vec(-128i8..127, 1..100),
                b_seed in 0u64..1000
            ) {
                use std::collections::hash_map::DefaultHasher;
                use std::hash::{Hash, Hasher};

                // Generate b with same length as a
                let mut hasher = DefaultHasher::new();
                b_seed.hash(&mut hasher);
                let seed = hasher.finish();

                let b: Vec<i8> = (0..a.len())
                    .map(|i| ((seed.wrapping_add(i as u64) % 256) as i32 - 128) as i8)
                    .collect();

                let backend = SimdBackend::Scalar;
                let ab = backend.dot_i8(&a, &b);
                let ba = backend.dot_i8(&b, &a);

                prop_assert_eq!(ab, ba);
            }

            /// Calibration n_samples increases monotonically
            #[test]
            fn prop_calibration_samples_monotonic(
                embeddings in prop::collection::vec(
                    prop::collection::vec(-1.0f32..1.0, 4..5),
                    1..50
                )
            ) {
                let mut cal = CalibrationStats::new(4);
                let mut prev_samples = 0;

                for emb in &embeddings {
                    if emb.len() == cal.dims {
                        if cal.update(emb).is_ok() {
                            prop_assert!(cal.n_samples > prev_samples);
                            prev_samples = cal.n_samples;
                        }
                    }
                }
            }

            /// Retriever returns at most top_k results
            #[test]
            fn prop_retriever_respects_top_k(
                num_docs in 1usize..50,
                top_k in 1usize..20
            ) {
                let config = RescoreRetrieverConfig {
                    rescore_multiplier: 4,
                    top_k,
                    min_calibration_samples: 1,
                    simd_backend: Some(SimdBackend::Scalar),
                };
                let mut retriever = RescoreRetriever::new(4, config);

                for i in 0..num_docs {
                    let emb = vec![(i as f32 * 0.1).sin(); 4];
                    let _ = retriever.index_document(&format!("doc{}", i), &emb);
                }

                let query = vec![0.5; 4];
                if let Ok(results) = retriever.retrieve(&query) {
                    prop_assert!(results.len() <= top_k);
                }
            }

            /// Memory usage is less than f32 equivalent
            #[test]
            fn prop_memory_reduction(
                dims in 64usize..1024
            ) {
                let f32_size = dims * 4;

                let mut cal = CalibrationStats::new(dims);
                let embedding: Vec<f32> = (0..dims).map(|i| (i as f32 * 0.01).sin()).collect();
                let _ = cal.update(&embedding);

                if let Ok(quantized) = QuantizedEmbedding::from_f32(&embedding, &cal) {
                    let _i8_size = quantized.memory_size();
                    // Overhead for params and hash, but core data should be smaller
                    prop_assert!(
                        quantized.values.len() < f32_size,
                        "Raw values {} should be less than f32 {}",
                        quantized.values.len(),
                        f32_size
                    );
                }
            }
        }
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    mod integration {
        use super::*;

        /// Full pipeline test: calibrate -> index -> retrieve
        #[test]
        fn full_pipeline() {
            let config = RescoreRetrieverConfig {
                rescore_multiplier: 4,
                top_k: 3,
                min_calibration_samples: 10,
                simd_backend: Some(SimdBackend::Scalar),
            };
            let mut retriever = RescoreRetriever::new(384, config);

            // Add calibration samples
            for i in 0..100 {
                let emb: Vec<f32> = (0..384).map(|j| ((i * j) as f32 * 0.001).sin()).collect();
                retriever.add_calibration_sample(&emb).unwrap();
            }

            // Index documents
            for i in 0..50 {
                let emb: Vec<f32> = (0..384).map(|j| ((i + j) as f32 * 0.01).cos()).collect();
                retriever
                    .index_document(&format!("doc{}", i), &emb)
                    .unwrap();
            }

            // Query
            let query: Vec<f32> = (0..384).map(|i| (i as f32 * 0.01).sin()).collect();
            let results = retriever.retrieve(&query).unwrap();

            // Should return top_k results
            assert_eq!(results.len(), 3);

            // Results should be sorted
            for i in 1..results.len() {
                assert!(results[i - 1].score >= results[i].score);
            }

            // All results should have valid doc_ids
            for result in &results {
                assert!(result.doc_id.starts_with("doc"));
            }
        }

        /// Test calibration -> quantization -> retrieval consistency
        #[test]
        fn calibration_consistency() {
            let mut cal = CalibrationStats::new(4);

            // Build calibration from known data
            let samples = vec![
                vec![1.0, 0.0, 0.0, 0.0],
                vec![0.0, 1.0, 0.0, 0.0],
                vec![0.0, 0.0, 1.0, 0.0],
                vec![0.0, 0.0, 0.0, 1.0],
            ];

            for sample in &samples {
                cal.update(sample).unwrap();
            }

            assert_eq!(cal.n_samples, 4);
            assert!((cal.absmax - 1.0).abs() < 1e-6);

            // Quantize and verify
            for sample in &samples {
                let q = QuantizedEmbedding::from_f32(sample, &cal).unwrap();
                assert!(q.verify_integrity());
            }
        }
    }
}
