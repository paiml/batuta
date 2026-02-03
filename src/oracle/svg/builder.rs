//! SVG Builder
//!
//! Fluent API for constructing SVG documents.

use super::layout::{LayoutEngine, Viewport, GRID_SIZE};
use super::palette::{Color, MaterialPalette, SovereignPalette};
use super::shapes::{ArrowMarker, Circle, Line, Path, Point, Rect, Text};
#[allow(unused_imports)]
pub use super::shapes::Size;
use super::typography::{MaterialTypography, TextStyle};

/// SVG element types
#[derive(Debug, Clone)]
pub enum SvgElement {
    Rect(Rect),
    Circle(Circle),
    Line(Line),
    Path(Path),
    Text(Text),
    Group { id: String, elements: Vec<SvgElement> },
}

impl SvgElement {
    /// Render to SVG string
    pub fn to_svg(&self) -> String {
        match self {
            Self::Rect(r) => r.to_svg(),
            Self::Circle(c) => c.to_svg(),
            Self::Line(l) => l.to_svg(),
            Self::Path(p) => p.to_svg(),
            Self::Text(t) => t.to_svg(),
            Self::Group { id, elements } => {
                let children: String = elements.iter().map(|e| e.to_svg()).collect();
                format!("<g id=\"{}\">{}</g>", id, children)
            }
        }
    }
}

/// SVG document builder
#[derive(Debug)]
pub struct SvgBuilder {
    /// Viewport dimensions
    viewport: Viewport,
    /// Material palette
    palette: MaterialPalette,
    /// Typography scale
    typography: MaterialTypography,
    /// Layout engine
    layout: LayoutEngine,
    /// SVG elements
    elements: Vec<SvgElement>,
    /// Marker definitions
    markers: Vec<ArrowMarker>,
    /// Custom CSS styles
    styles: Vec<String>,
    /// SVG title
    title: Option<String>,
    /// SVG description
    description: Option<String>,
}

impl SvgBuilder {
    /// Create a new SVG builder with default settings
    pub fn new() -> Self {
        let viewport = Viewport::presentation();
        let palette = MaterialPalette::light();
        let typography = MaterialTypography::with_color(palette.on_surface);

        Self {
            viewport,
            palette: palette.clone(),
            typography,
            layout: LayoutEngine::new(viewport),
            elements: Vec::new(),
            markers: Vec::new(),
            styles: Vec::new(),
            title: None,
            description: None,
        }
    }

    /// Set the viewport
    pub fn viewport(mut self, viewport: Viewport) -> Self {
        self.viewport = viewport;
        self.layout = LayoutEngine::new(viewport);
        self
    }

    /// Set the viewport size
    pub fn size(self, width: f32, height: f32) -> Self {
        self.viewport(Viewport::new(width, height))
    }

    /// Use the document viewport (800x600)
    pub fn document(self) -> Self {
        self.viewport(Viewport::document())
    }

    /// Use the presentation viewport (1920x1080)
    pub fn presentation(self) -> Self {
        self.viewport(Viewport::presentation())
    }

    /// Set the color palette
    pub fn palette(mut self, palette: MaterialPalette) -> Self {
        self.typography = MaterialTypography::with_color(palette.on_surface);
        self.palette = palette;
        self
    }

    /// Use the dark palette
    pub fn dark_mode(self) -> Self {
        self.palette(MaterialPalette::dark())
    }

    /// Set the title
    pub fn title(mut self, title: &str) -> Self {
        self.title = Some(title.to_string());
        self
    }

    /// Set the description
    pub fn description(mut self, desc: &str) -> Self {
        self.description = Some(desc.to_string());
        self
    }

    /// Add a custom CSS style
    pub fn add_style(mut self, css: &str) -> Self {
        self.styles.push(css.to_string());
        self
    }

    /// Add a rectangle
    pub fn rect(mut self, id: &str, x: f32, y: f32, width: f32, height: f32) -> Self {
        let rect = Rect::new(x, y, width, height)
            .with_fill(self.palette.surface)
            .with_stroke(self.palette.outline, 1.0);

        if self.layout.add(id, rect.clone()) {
            self.elements.push(SvgElement::Rect(rect));
        }
        self
    }

    /// Add a styled rectangle
    pub fn rect_styled(
        mut self,
        id: &str,
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        fill: Color,
        stroke: Option<(Color, f32)>,
        radius: f32,
    ) -> Self {
        let mut rect = Rect::new(x, y, width, height)
            .with_fill(fill)
            .with_radius(radius);

        if let Some((color, width)) = stroke {
            rect = rect.with_stroke(color, width);
        }

        if self.layout.add(id, rect.clone()) {
            self.elements.push(SvgElement::Rect(rect));
        }
        self
    }

    /// Add a circle
    pub fn circle(mut self, id: &str, cx: f32, cy: f32, r: f32) -> Self {
        let circle = Circle::new(cx, cy, r)
            .with_fill(self.palette.primary)
            .with_stroke(self.palette.outline, 1.0);

        let bounds = circle.bounds();
        if self.layout.add(id, bounds) {
            self.elements.push(SvgElement::Circle(circle));
        }
        self
    }

    /// Add a styled circle
    pub fn circle_styled(
        mut self,
        id: &str,
        cx: f32,
        cy: f32,
        r: f32,
        fill: Color,
        stroke: Option<(Color, f32)>,
    ) -> Self {
        let mut circle = Circle::new(cx, cy, r).with_fill(fill);

        if let Some((color, width)) = stroke {
            circle = circle.with_stroke(color, width);
        }

        let bounds = circle.bounds();
        if self.layout.add(id, bounds) {
            self.elements.push(SvgElement::Circle(circle));
        }
        self
    }

    /// Add a line
    pub fn line(mut self, x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        let line = Line::new(x1, y1, x2, y2).with_stroke(self.palette.outline);
        self.elements.push(SvgElement::Line(line));
        self
    }

    /// Add a styled line
    pub fn line_styled(
        mut self,
        x1: f32,
        y1: f32,
        x2: f32,
        y2: f32,
        color: Color,
        width: f32,
    ) -> Self {
        let line = Line::new(x1, y1, x2, y2)
            .with_stroke(color)
            .with_stroke_width(width);
        self.elements.push(SvgElement::Line(line));
        self
    }

    /// Add text
    pub fn text(mut self, x: f32, y: f32, content: &str) -> Self {
        let text = Text::new(x, y, content).with_style(self.typography.body_medium.clone());
        self.elements.push(SvgElement::Text(text));
        self
    }

    /// Add text with a specific style
    pub fn text_styled(mut self, x: f32, y: f32, content: &str, style: TextStyle) -> Self {
        let text = Text::new(x, y, content).with_style(style);
        self.elements.push(SvgElement::Text(text));
        self
    }

    /// Add a title (headline style)
    pub fn heading(mut self, x: f32, y: f32, content: &str) -> Self {
        let text = Text::new(x, y, content).with_style(self.typography.headline_medium.clone());
        self.elements.push(SvgElement::Text(text));
        self
    }

    /// Add a label (small text)
    pub fn label(mut self, x: f32, y: f32, content: &str) -> Self {
        let text = Text::new(x, y, content).with_style(self.typography.label_medium.clone());
        self.elements.push(SvgElement::Text(text));
        self
    }

    /// Add a path
    pub fn path(mut self, path: Path) -> Self {
        self.elements.push(SvgElement::Path(path));
        self
    }

    /// Add an arrow marker definition
    pub fn add_arrow_marker(mut self, id: &str, color: Color) -> Self {
        self.markers.push(ArrowMarker::new(id, color));
        self
    }

    /// Add an element directly
    pub fn element(mut self, element: SvgElement) -> Self {
        self.elements.push(element);
        self
    }

    /// Add a group of elements
    pub fn group(mut self, id: &str, elements: Vec<SvgElement>) -> Self {
        self.elements.push(SvgElement::Group {
            id: id.to_string(),
            elements,
        });
        self
    }

    /// Get the palette
    pub fn get_palette(&self) -> &MaterialPalette {
        &self.palette
    }

    /// Get the typography
    pub fn get_typography(&self) -> &MaterialTypography {
        &self.typography
    }

    /// Get the layout engine
    pub fn get_layout(&self) -> &LayoutEngine {
        &self.layout
    }

    /// Get the layout engine mutably
    pub fn get_layout_mut(&mut self) -> &mut LayoutEngine {
        &mut self.layout
    }

    /// Validate the SVG layout
    pub fn validate(&self) -> Vec<String> {
        let mut errors = Vec::new();

        // Check layout errors
        for error in self.layout.validate() {
            errors.push(error.to_string());
        }

        // Check for valid palette colors (simplified check)
        // In production, would validate all element colors

        errors
    }

    /// Estimate the output file size
    pub fn estimate_size(&self) -> usize {
        // Rough estimate: base overhead + per-element overhead
        let base = 500; // XML declaration, root element, etc.
        let per_element = 100; // Average element size
        let marker_overhead = self.markers.len() * 200;
        let style_overhead: usize = self.styles.iter().map(|s| s.len()).sum();

        base + self.elements.len() * per_element + marker_overhead + style_overhead
    }

    /// Build the SVG document
    pub fn build(self) -> String {
        let mut svg = String::new();

        // XML declaration
        svg.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");

        // SVG root element
        svg.push_str(&format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"{}\" width=\"{}\" height=\"{}\">\n",
            self.viewport.view_box(),
            self.viewport.width,
            self.viewport.height
        ));

        // Title and description
        if let Some(title) = &self.title {
            svg.push_str(&format!("  <title>{}</title>\n", title));
        }
        if let Some(desc) = &self.description {
            svg.push_str(&format!("  <desc>{}</desc>\n", desc));
        }

        // Styles
        if !self.styles.is_empty() {
            svg.push_str("  <style>\n");
            for style in &self.styles {
                svg.push_str(&format!("    {}\n", style));
            }
            svg.push_str("  </style>\n");
        }

        // Definitions (markers)
        if !self.markers.is_empty() {
            svg.push_str("  <defs>\n");
            for marker in &self.markers {
                svg.push_str(&format!("    {}\n", marker.to_svg_def()));
            }
            svg.push_str("  </defs>\n");
        }

        // Background
        svg.push_str(&format!(
            "  <rect width=\"100%\" height=\"100%\" fill=\"{}\"/>\n",
            self.palette.background.to_css_hex()
        ));

        // Elements
        for element in &self.elements {
            svg.push_str(&format!("  {}\n", element.to_svg()));
        }

        svg.push_str("</svg>\n");

        svg
    }
}

impl Default for SvgBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for component diagrams (shape-heavy)
pub struct ComponentDiagramBuilder {
    builder: SvgBuilder,
    palette: SovereignPalette,
}

impl ComponentDiagramBuilder {
    /// Create a new component diagram builder
    pub fn new() -> Self {
        Self {
            builder: SvgBuilder::new().presentation(),
            palette: SovereignPalette::light(),
        }
    }

    /// Add a component box
    pub fn component(mut self, id: &str, x: f32, y: f32, name: &str, component_type: &str) -> Self {
        let width = 160.0;
        let height = 80.0;
        let color = self.palette.component_color(component_type);

        self.builder = self.builder.rect_styled(
            id,
            x,
            y,
            width,
            height,
            color.lighten(0.8),
            Some((color, 2.0)),
            GRID_SIZE,
        );

        // Component label
        let text_style = self
            .builder
            .get_typography()
            .title_small
            .clone()
            .with_color(self.palette.material.on_surface);
        self.builder = self
            .builder
            .text_styled(x + width / 2.0, y + height / 2.0 + 5.0, name, text_style);

        self
    }

    /// Add a connection arrow
    pub fn connect(mut self, from: Point, to: Point) -> Self {
        self.builder = self.builder.line_styled(
            from.x,
            from.y,
            to.x,
            to.y,
            self.palette.material.outline,
            2.0,
        );
        self
    }

    /// Build the diagram
    pub fn build(self) -> String {
        self.builder.build()
    }
}

impl Default for ComponentDiagramBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_svg_builder_creation() {
        let builder = SvgBuilder::new();
        assert_eq!(builder.viewport.width, 1920.0);
        assert_eq!(builder.viewport.height, 1080.0);
    }

    #[test]
    fn test_svg_builder_viewport() {
        let builder = SvgBuilder::new().document();
        assert_eq!(builder.viewport.width, 800.0);
        assert_eq!(builder.viewport.height, 600.0);
    }

    #[test]
    fn test_svg_builder_rect() {
        let svg = SvgBuilder::new()
            .size(200.0, 200.0)
            .rect("test", 10.0, 10.0, 50.0, 50.0)
            .build();

        assert!(svg.contains("<rect"));
        assert!(svg.contains("width=\"50\""));
    }

    #[test]
    fn test_svg_builder_circle() {
        let svg = SvgBuilder::new()
            .size(200.0, 200.0)
            .circle("test", 50.0, 50.0, 25.0)
            .build();

        assert!(svg.contains("<circle"));
        assert!(svg.contains("r=\"25\""));
    }

    #[test]
    fn test_svg_builder_text() {
        let svg = SvgBuilder::new()
            .size(200.0, 200.0)
            .text(10.0, 20.0, "Hello")
            .build();

        assert!(svg.contains("<text"));
        assert!(svg.contains("Hello"));
    }

    #[test]
    fn test_svg_builder_title() {
        let svg = SvgBuilder::new()
            .title("Test Diagram")
            .description("A test")
            .build();

        assert!(svg.contains("<title>Test Diagram</title>"));
        assert!(svg.contains("<desc>A test</desc>"));
    }

    #[test]
    fn test_svg_builder_dark_mode() {
        let builder = SvgBuilder::new().dark_mode();
        assert_eq!(
            builder.palette.surface.to_css_hex(),
            MaterialPalette::dark().surface.to_css_hex()
        );
    }

    #[test]
    fn test_svg_builder_validation() {
        // Use a larger viewport with no padding issues
        let builder = SvgBuilder::new()
            .size(200.0, 200.0)
            .rect("r1", 24.0, 24.0, 48.0, 48.0); // Use grid-aligned values inside content area

        let errors = builder.validate();
        // Should be valid (properly aligned and within bounds)
        assert!(errors.is_empty(), "Unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_svg_builder_estimate_size() {
        let builder = SvgBuilder::new()
            .rect("r1", 0.0, 0.0, 50.0, 50.0)
            .rect("r2", 60.0, 0.0, 50.0, 50.0);

        let size = builder.estimate_size();
        assert!(size > 0);
        assert!(size < 10000); // Should be under 10KB
    }

    #[test]
    fn test_svg_element_group() {
        let group = SvgElement::Group {
            id: "test-group".to_string(),
            elements: vec![
                SvgElement::Rect(Rect::new(0.0, 0.0, 10.0, 10.0)),
                SvgElement::Circle(Circle::new(5.0, 5.0, 2.0)),
            ],
        };

        let svg = group.to_svg();
        assert!(svg.contains("id=\"test-group\""));
        assert!(svg.contains("<rect"));
        assert!(svg.contains("<circle"));
    }

    #[test]
    fn test_component_diagram_builder() {
        let svg = ComponentDiagramBuilder::new()
            .component("trueno", 100.0, 100.0, "Trueno", "trueno")
            .component("aprender", 300.0, 100.0, "Aprender", "aprender")
            .connect(Point::new(260.0, 140.0), Point::new(300.0, 140.0))
            .build();

        assert!(svg.contains("<svg"));
        assert!(svg.contains("Trueno"));
        assert!(svg.contains("Aprender"));
    }

    #[test]
    fn test_svg_output_size() {
        let svg = SvgBuilder::new()
            .size(800.0, 600.0)
            .title("Test")
            .rect("r1", 10.0, 10.0, 100.0, 100.0)
            .circle("c1", 200.0, 200.0, 30.0)
            .text(50.0, 50.0, "Hello World")
            .build();

        // Output should be under 100KB as per spec
        assert!(svg.len() < 100_000, "SVG too large: {} bytes", svg.len());
    }
}
