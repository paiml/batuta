//! Material Design 3 Color Palette
//!
//! Defines color schemes based on Material Design 3 specification.

use std::fmt;

/// An RGBA color
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub r: u8,
    pub g: u8,
    pub b: u8,
    pub a: u8,
}

impl Color {
    /// Create a new color with full opacity
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// Create a new color with custom opacity
    pub const fn rgba(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create from hex string (e.g., "#6750A4" or "6750A4")
    pub fn from_hex(hex: &str) -> Option<Self> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            return None;
        }

        let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
        let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
        let b = u8::from_str_radix(&hex[4..6], 16).ok()?;

        Some(Self::rgb(r, g, b))
    }

    /// Convert to hex string (without #)
    pub fn to_hex(&self) -> String {
        format!("{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to CSS hex string (with #)
    pub fn to_css_hex(&self) -> String {
        format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
    }

    /// Convert to CSS rgba string
    pub fn to_css_rgba(&self) -> String {
        if self.a == 255 {
            format!("rgb({}, {}, {})", self.r, self.g, self.b)
        } else {
            format!(
                "rgba({}, {}, {}, {:.2})",
                self.r,
                self.g,
                self.b,
                self.a as f32 / 255.0
            )
        }
    }

    /// Apply opacity (0.0 - 1.0) to this color
    pub fn with_opacity(&self, opacity: f32) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a: (opacity.clamp(0.0, 1.0) * 255.0) as u8,
        }
    }

    /// Lighten the color by a percentage (0.0 - 1.0)
    pub fn lighten(&self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        Self {
            r: (self.r as f32 + (255.0 - self.r as f32) * amount) as u8,
            g: (self.g as f32 + (255.0 - self.g as f32) * amount) as u8,
            b: (self.b as f32 + (255.0 - self.b as f32) * amount) as u8,
            a: self.a,
        }
    }

    /// Darken the color by a percentage (0.0 - 1.0)
    pub fn darken(&self, amount: f32) -> Self {
        let amount = amount.clamp(0.0, 1.0);
        Self {
            r: (self.r as f32 * (1.0 - amount)) as u8,
            g: (self.g as f32 * (1.0 - amount)) as u8,
            b: (self.b as f32 * (1.0 - amount)) as u8,
            a: self.a,
        }
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_css_hex())
    }
}

impl Default for Color {
    fn default() -> Self {
        Self::rgb(0, 0, 0)
    }
}

/// Material Design 3 color palette
#[derive(Debug, Clone)]
pub struct MaterialPalette {
    /// Primary color - #6750A4
    pub primary: Color,
    /// On-primary text color
    pub on_primary: Color,
    /// Primary container
    pub primary_container: Color,
    /// On-primary container text
    pub on_primary_container: Color,

    /// Secondary color
    pub secondary: Color,
    /// On-secondary text color
    pub on_secondary: Color,

    /// Tertiary color
    pub tertiary: Color,
    /// On-tertiary text color
    pub on_tertiary: Color,

    /// Error color
    pub error: Color,
    /// On-error text color
    pub on_error: Color,

    /// Surface color - #FFFBFE
    pub surface: Color,
    /// On-surface text color
    pub on_surface: Color,
    /// Surface variant
    pub surface_variant: Color,
    /// On-surface variant text
    pub on_surface_variant: Color,

    /// Outline color - #79747E
    pub outline: Color,
    /// Outline variant (lighter)
    pub outline_variant: Color,

    /// Background color
    pub background: Color,
    /// On-background text color
    pub on_background: Color,
}

impl MaterialPalette {
    /// Create the default Material Design 3 light palette
    pub fn light() -> Self {
        Self {
            // Primary (Purple)
            primary: Color::from_hex("#6750A4").unwrap(),
            on_primary: Color::rgb(255, 255, 255),
            primary_container: Color::from_hex("#EADDFF").unwrap(),
            on_primary_container: Color::from_hex("#21005D").unwrap(),

            // Secondary (Pink-purple)
            secondary: Color::from_hex("#625B71").unwrap(),
            on_secondary: Color::rgb(255, 255, 255),

            // Tertiary (Teal)
            tertiary: Color::from_hex("#7D5260").unwrap(),
            on_tertiary: Color::rgb(255, 255, 255),

            // Error (Red)
            error: Color::from_hex("#B3261E").unwrap(),
            on_error: Color::rgb(255, 255, 255),

            // Surface
            surface: Color::from_hex("#FFFBFE").unwrap(),
            on_surface: Color::from_hex("#1C1B1F").unwrap(),
            surface_variant: Color::from_hex("#E7E0EC").unwrap(),
            on_surface_variant: Color::from_hex("#49454F").unwrap(),

            // Outline
            outline: Color::from_hex("#79747E").unwrap(),
            outline_variant: Color::from_hex("#CAC4D0").unwrap(),

            // Background
            background: Color::from_hex("#FFFBFE").unwrap(),
            on_background: Color::from_hex("#1C1B1F").unwrap(),
        }
    }

    /// Create the Material Design 3 dark palette
    pub fn dark() -> Self {
        Self {
            // Primary (Purple)
            primary: Color::from_hex("#D0BCFF").unwrap(),
            on_primary: Color::from_hex("#381E72").unwrap(),
            primary_container: Color::from_hex("#4F378B").unwrap(),
            on_primary_container: Color::from_hex("#EADDFF").unwrap(),

            // Secondary
            secondary: Color::from_hex("#CCC2DC").unwrap(),
            on_secondary: Color::from_hex("#332D41").unwrap(),

            // Tertiary
            tertiary: Color::from_hex("#EFB8C8").unwrap(),
            on_tertiary: Color::from_hex("#492532").unwrap(),

            // Error
            error: Color::from_hex("#F2B8B5").unwrap(),
            on_error: Color::from_hex("#601410").unwrap(),

            // Surface
            surface: Color::from_hex("#1C1B1F").unwrap(),
            on_surface: Color::from_hex("#E6E1E5").unwrap(),
            surface_variant: Color::from_hex("#49454F").unwrap(),
            on_surface_variant: Color::from_hex("#CAC4D0").unwrap(),

            // Outline
            outline: Color::from_hex("#938F99").unwrap(),
            outline_variant: Color::from_hex("#49454F").unwrap(),

            // Background
            background: Color::from_hex("#1C1B1F").unwrap(),
            on_background: Color::from_hex("#E6E1E5").unwrap(),
        }
    }

    /// Create a custom palette with a primary color
    pub fn with_primary(primary: Color) -> Self {
        let mut palette = Self::light();
        palette.primary = primary;
        palette
    }

    /// Validate that a color is in this palette
    pub fn is_valid_color(&self, color: &Color) -> bool {
        color == &self.primary
            || color == &self.on_primary
            || color == &self.primary_container
            || color == &self.on_primary_container
            || color == &self.secondary
            || color == &self.on_secondary
            || color == &self.tertiary
            || color == &self.on_tertiary
            || color == &self.error
            || color == &self.on_error
            || color == &self.surface
            || color == &self.on_surface
            || color == &self.surface_variant
            || color == &self.on_surface_variant
            || color == &self.outline
            || color == &self.outline_variant
            || color == &self.background
            || color == &self.on_background
    }

    /// Get all colors in the palette
    pub fn all_colors(&self) -> Vec<Color> {
        vec![
            self.primary,
            self.on_primary,
            self.primary_container,
            self.on_primary_container,
            self.secondary,
            self.on_secondary,
            self.tertiary,
            self.on_tertiary,
            self.error,
            self.on_error,
            self.surface,
            self.on_surface,
            self.surface_variant,
            self.on_surface_variant,
            self.outline,
            self.outline_variant,
            self.background,
            self.on_background,
        ]
    }
}

impl Default for MaterialPalette {
    fn default() -> Self {
        Self::light()
    }
}

/// Sovereign AI Stack color scheme (extends Material Design 3)
#[derive(Debug, Clone)]
pub struct SovereignPalette {
    /// Base Material palette
    pub material: MaterialPalette,

    /// Trueno component color (orange)
    pub trueno: Color,
    /// Aprender component color (blue)
    pub aprender: Color,
    /// Realizar component color (green)
    pub realizar: Color,
    /// Batuta component color (purple)
    pub batuta: Color,

    /// Success color
    pub success: Color,
    /// Warning color
    pub warning: Color,
    /// Info color
    pub info: Color,
}

impl SovereignPalette {
    /// Create the light sovereign palette
    pub fn light() -> Self {
        Self {
            material: MaterialPalette::light(),
            trueno: Color::from_hex("#FF6D00").unwrap(),   // Deep Orange A400
            aprender: Color::from_hex("#2962FF").unwrap(), // Blue A700
            realizar: Color::from_hex("#00C853").unwrap(), // Green A700
            batuta: Color::from_hex("#6750A4").unwrap(),   // Primary Purple
            success: Color::from_hex("#00C853").unwrap(),  // Green A700
            warning: Color::from_hex("#FFD600").unwrap(),  // Yellow A700
            info: Color::from_hex("#00B0FF").unwrap(),     // Light Blue A400
        }
    }

    /// Create the dark sovereign palette
    pub fn dark() -> Self {
        Self {
            material: MaterialPalette::dark(),
            trueno: Color::from_hex("#FFAB40").unwrap(),   // Orange A200
            aprender: Color::from_hex("#82B1FF").unwrap(), // Blue A100
            realizar: Color::from_hex("#69F0AE").unwrap(), // Green A200
            batuta: Color::from_hex("#D0BCFF").unwrap(),   // Primary Purple
            success: Color::from_hex("#69F0AE").unwrap(),  // Green A200
            warning: Color::from_hex("#FFE57F").unwrap(),  // Amber A200
            info: Color::from_hex("#80D8FF").unwrap(),     // Light Blue A100
        }
    }

    /// Get color for a stack component
    pub fn component_color(&self, component: &str) -> Color {
        match component.to_lowercase().as_str() {
            "trueno" => self.trueno,
            "aprender" => self.aprender,
            "realizar" => self.realizar,
            "batuta" => self.batuta,
            _ => self.material.outline,
        }
    }
}

impl Default for SovereignPalette {
    fn default() -> Self {
        Self::light()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_from_hex() {
        let color = Color::from_hex("#6750A4").unwrap();
        assert_eq!(color.r, 103);
        assert_eq!(color.g, 80);
        assert_eq!(color.b, 164);
    }

    #[test]
    fn test_color_from_hex_no_hash() {
        let color = Color::from_hex("6750A4").unwrap();
        assert_eq!(color.r, 103);
        assert_eq!(color.g, 80);
        assert_eq!(color.b, 164);
    }

    #[test]
    fn test_color_to_hex() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(color.to_hex(), "6750A4");
        assert_eq!(color.to_css_hex(), "#6750A4");
    }

    #[test]
    fn test_color_to_css_rgba() {
        let color = Color::rgb(103, 80, 164);
        assert_eq!(color.to_css_rgba(), "rgb(103, 80, 164)");

        let color_alpha = Color::rgba(103, 80, 164, 128);
        assert!(color_alpha.to_css_rgba().starts_with("rgba(103, 80, 164,"));
    }

    #[test]
    fn test_color_with_opacity() {
        let color = Color::rgb(255, 255, 255);
        let semi = color.with_opacity(0.5);
        assert_eq!(semi.a, 127);
    }

    #[test]
    fn test_color_lighten() {
        let color = Color::rgb(100, 100, 100);
        let lighter = color.lighten(0.5);
        assert!(lighter.r > color.r);
        assert!(lighter.g > color.g);
        assert!(lighter.b > color.b);
    }

    #[test]
    fn test_color_darken() {
        let color = Color::rgb(100, 100, 100);
        let darker = color.darken(0.5);
        assert!(darker.r < color.r);
        assert!(darker.g < color.g);
        assert!(darker.b < color.b);
    }

    #[test]
    fn test_material_palette_light() {
        let palette = MaterialPalette::light();
        assert_eq!(palette.primary.to_css_hex(), "#6750A4");
        assert_eq!(palette.surface.to_css_hex(), "#FFFBFE");
        assert_eq!(palette.outline.to_css_hex(), "#79747E");
    }

    #[test]
    fn test_material_palette_dark() {
        let palette = MaterialPalette::dark();
        assert_eq!(palette.primary.to_css_hex(), "#D0BCFF");
        assert_eq!(palette.surface.to_css_hex(), "#1C1B1F");
    }

    #[test]
    fn test_material_palette_validation() {
        let palette = MaterialPalette::light();
        assert!(palette.is_valid_color(&palette.primary));
        assert!(palette.is_valid_color(&palette.surface));
        assert!(!palette.is_valid_color(&Color::rgb(1, 2, 3)));
    }

    #[test]
    fn test_sovereign_palette() {
        let palette = SovereignPalette::light();
        assert_eq!(palette.trueno.to_css_hex(), "#FF6D00");
        assert_eq!(palette.aprender.to_css_hex(), "#2962FF");
        assert_eq!(palette.realizar.to_css_hex(), "#00C853");
    }

    #[test]
    fn test_sovereign_component_color() {
        let palette = SovereignPalette::light();
        assert_eq!(palette.component_color("trueno"), palette.trueno);
        assert_eq!(palette.component_color("APRENDER"), palette.aprender);
        assert_eq!(palette.component_color("unknown"), palette.material.outline);
    }
}
