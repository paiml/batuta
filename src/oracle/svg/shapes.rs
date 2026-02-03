//! SVG Shape Primitives
//!
//! Basic shapes for building diagrams: rectangles, circles, paths, text.

use super::palette::Color;
use super::typography::TextStyle;

/// Point in 2D space
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Distance to another point
    pub fn distance(&self, other: &Point) -> f32 {
        ((self.x - other.x).powi(2) + (self.y - other.y).powi(2)).sqrt()
    }

    /// Midpoint between two points
    pub fn midpoint(&self, other: &Point) -> Point {
        Point::new((self.x + other.x) / 2.0, (self.y + other.y) / 2.0)
    }
}

/// Size (width and height)
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Size {
    pub width: f32,
    pub height: f32,
}

impl Size {
    pub const fn new(width: f32, height: f32) -> Self {
        Self { width, height }
    }

    /// Area
    pub fn area(&self) -> f32 {
        self.width * self.height
    }
}

/// A rectangle
#[derive(Debug, Clone, PartialEq)]
pub struct Rect {
    /// Top-left corner position
    pub position: Point,
    /// Size
    pub size: Size,
    /// Corner radius (0 for sharp corners)
    pub corner_radius: f32,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Rect {
    /// Create a new rectangle
    pub fn new(x: f32, y: f32, width: f32, height: f32) -> Self {
        Self {
            position: Point::new(x, y),
            size: Size::new(width, height),
            corner_radius: 0.0,
            fill: None,
            stroke: None,
            stroke_width: 1.0,
        }
    }

    /// Set corner radius
    pub fn with_radius(mut self, radius: f32) -> Self {
        self.corner_radius = radius;
        self
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get center point
    pub fn center(&self) -> Point {
        Point::new(
            self.position.x + self.size.width / 2.0,
            self.position.y + self.size.height / 2.0,
        )
    }

    /// Get right edge x coordinate
    pub fn right(&self) -> f32 {
        self.position.x + self.size.width
    }

    /// Get bottom edge y coordinate
    pub fn bottom(&self) -> f32 {
        self.position.y + self.size.height
    }

    /// Check if a point is inside the rectangle
    pub fn contains(&self, point: &Point) -> bool {
        point.x >= self.position.x
            && point.x <= self.right()
            && point.y >= self.position.y
            && point.y <= self.bottom()
    }

    /// Check if two rectangles overlap
    pub fn intersects(&self, other: &Rect) -> bool {
        self.position.x < other.right()
            && self.right() > other.position.x
            && self.position.y < other.bottom()
            && self.bottom() > other.position.y
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!(
            "x=\"{}\" y=\"{}\" width=\"{}\" height=\"{}\"",
            self.position.x, self.position.y, self.size.width, self.size.height
        );

        if self.corner_radius > 0.0 {
            attrs.push_str(&format!(" rx=\"{}\"", self.corner_radius));
        }

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<rect {}/>", attrs)
    }
}

impl Default for Rect {
    fn default() -> Self {
        Self::new(0.0, 0.0, 100.0, 100.0)
    }
}

/// A circle
#[derive(Debug, Clone, PartialEq)]
pub struct Circle {
    /// Center position
    pub center: Point,
    /// Radius
    pub radius: f32,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Circle {
    /// Create a new circle
    pub fn new(cx: f32, cy: f32, r: f32) -> Self {
        Self {
            center: Point::new(cx, cy),
            radius: r,
            fill: None,
            stroke: None,
            stroke_width: 1.0,
        }
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get bounding rectangle
    pub fn bounds(&self) -> Rect {
        Rect::new(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.radius * 2.0,
            self.radius * 2.0,
        )
    }

    /// Check if a point is inside the circle
    pub fn contains(&self, point: &Point) -> bool {
        self.center.distance(point) <= self.radius
    }

    /// Check if two circles overlap
    pub fn intersects(&self, other: &Circle) -> bool {
        self.center.distance(&other.center) < self.radius + other.radius
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!(
            "cx=\"{}\" cy=\"{}\" r=\"{}\"",
            self.center.x, self.center.y, self.radius
        );

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<circle {}/>", attrs)
    }
}

impl Default for Circle {
    fn default() -> Self {
        Self::new(50.0, 50.0, 25.0)
    }
}

/// A line segment
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    /// Start point
    pub start: Point,
    /// End point
    pub end: Point,
    /// Stroke color
    pub stroke: Color,
    /// Stroke width
    pub stroke_width: f32,
    /// Dash array (for dashed lines)
    pub dash_array: Option<String>,
}

impl Line {
    /// Create a new line
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32) -> Self {
        Self {
            start: Point::new(x1, y1),
            end: Point::new(x2, y2),
            stroke: Color::rgb(0, 0, 0),
            stroke_width: 1.0,
            dash_array: None,
        }
    }

    /// Set stroke color
    pub fn with_stroke(mut self, color: Color) -> Self {
        self.stroke = color;
        self
    }

    /// Set stroke width
    pub fn with_stroke_width(mut self, width: f32) -> Self {
        self.stroke_width = width;
        self
    }

    /// Set dash pattern
    pub fn with_dash(mut self, pattern: &str) -> Self {
        self.dash_array = Some(pattern.to_string());
        self
    }

    /// Get the length of the line
    pub fn length(&self) -> f32 {
        self.start.distance(&self.end)
    }

    /// Get the midpoint
    pub fn midpoint(&self) -> Point {
        self.start.midpoint(&self.end)
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!(
            "x1=\"{}\" y1=\"{}\" x2=\"{}\" y2=\"{}\" stroke=\"{}\" stroke-width=\"{}\"",
            self.start.x,
            self.start.y,
            self.end.x,
            self.end.y,
            self.stroke.to_css_hex(),
            self.stroke_width
        );

        if let Some(dash) = &self.dash_array {
            attrs.push_str(&format!(" stroke-dasharray=\"{}\"", dash));
        }

        format!("<line {}/>", attrs)
    }
}

/// SVG path commands
#[derive(Debug, Clone)]
pub enum PathCommand {
    /// Move to (x, y)
    MoveTo(f32, f32),
    /// Line to (x, y)
    LineTo(f32, f32),
    /// Horizontal line to x
    HorizontalTo(f32),
    /// Vertical line to y
    VerticalTo(f32),
    /// Quadratic curve to (x, y) with control point (cx, cy)
    QuadraticTo { cx: f32, cy: f32, x: f32, y: f32 },
    /// Cubic curve to (x, y) with control points
    CubicTo {
        cx1: f32,
        cy1: f32,
        cx2: f32,
        cy2: f32,
        x: f32,
        y: f32,
    },
    /// Arc to (x, y)
    ArcTo {
        rx: f32,
        ry: f32,
        rotation: f32,
        large_arc: bool,
        sweep: bool,
        x: f32,
        y: f32,
    },
    /// Close path
    Close,
}

impl PathCommand {
    /// Convert to SVG path data string
    pub fn to_svg(&self) -> String {
        match self {
            Self::MoveTo(x, y) => format!("M {} {}", x, y),
            Self::LineTo(x, y) => format!("L {} {}", x, y),
            Self::HorizontalTo(x) => format!("H {}", x),
            Self::VerticalTo(y) => format!("V {}", y),
            Self::QuadraticTo { cx, cy, x, y } => format!("Q {} {} {} {}", cx, cy, x, y),
            Self::CubicTo {
                cx1,
                cy1,
                cx2,
                cy2,
                x,
                y,
            } => format!("C {} {} {} {} {} {}", cx1, cy1, cx2, cy2, x, y),
            Self::ArcTo {
                rx,
                ry,
                rotation,
                large_arc,
                sweep,
                x,
                y,
            } => format!(
                "A {} {} {} {} {} {} {}",
                rx,
                ry,
                rotation,
                if *large_arc { 1 } else { 0 },
                if *sweep { 1 } else { 0 },
                x,
                y
            ),
            Self::Close => "Z".to_string(),
        }
    }
}

/// A path shape
#[derive(Debug, Clone)]
pub struct Path {
    /// Path commands
    pub commands: Vec<PathCommand>,
    /// Fill color
    pub fill: Option<Color>,
    /// Stroke color
    pub stroke: Option<Color>,
    /// Stroke width
    pub stroke_width: f32,
}

impl Path {
    /// Create a new empty path
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            fill: None,
            stroke: None,
            stroke_width: 1.0,
        }
    }

    /// Move to a point
    pub fn move_to(mut self, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::MoveTo(x, y));
        self
    }

    /// Line to a point
    pub fn line_to(mut self, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::LineTo(x, y));
        self
    }

    /// Quadratic curve to a point
    pub fn quad_to(mut self, cx: f32, cy: f32, x: f32, y: f32) -> Self {
        self.commands
            .push(PathCommand::QuadraticTo { cx, cy, x, y });
        self
    }

    /// Cubic curve to a point
    pub fn cubic_to(mut self, cx1: f32, cy1: f32, cx2: f32, cy2: f32, x: f32, y: f32) -> Self {
        self.commands.push(PathCommand::CubicTo {
            cx1,
            cy1,
            cx2,
            cy2,
            x,
            y,
        });
        self
    }

    /// Close the path
    pub fn close(mut self) -> Self {
        self.commands.push(PathCommand::Close);
        self
    }

    /// Set fill color
    pub fn with_fill(mut self, color: Color) -> Self {
        self.fill = Some(color);
        self
    }

    /// Set stroke
    pub fn with_stroke(mut self, color: Color, width: f32) -> Self {
        self.stroke = Some(color);
        self.stroke_width = width;
        self
    }

    /// Get the path data string
    pub fn to_path_data(&self) -> String {
        self.commands
            .iter()
            .map(|c| c.to_svg())
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let mut attrs = format!("d=\"{}\"", self.to_path_data());

        if let Some(fill) = &self.fill {
            attrs.push_str(&format!(" fill=\"{}\"", fill.to_css_hex()));
        } else {
            attrs.push_str(" fill=\"none\"");
        }

        if let Some(stroke) = &self.stroke {
            attrs.push_str(&format!(
                " stroke=\"{}\" stroke-width=\"{}\"",
                stroke.to_css_hex(),
                self.stroke_width
            ));
        }

        format!("<path {}/>", attrs)
    }
}

impl Default for Path {
    fn default() -> Self {
        Self::new()
    }
}

/// A text element
#[derive(Debug, Clone)]
pub struct Text {
    /// Position
    pub position: Point,
    /// Text content
    pub content: String,
    /// Style
    pub style: TextStyle,
}

impl Text {
    /// Create a new text element
    pub fn new(x: f32, y: f32, content: &str) -> Self {
        Self {
            position: Point::new(x, y),
            content: content.to_string(),
            style: TextStyle::default(),
        }
    }

    /// Set the text style
    pub fn with_style(mut self, style: TextStyle) -> Self {
        self.style = style;
        self
    }

    /// Render to SVG element
    pub fn to_svg(&self) -> String {
        let style_attrs = self.style.to_svg_attrs();
        format!(
            "<text x=\"{}\" y=\"{}\" {}>{}</text>",
            self.position.x,
            self.position.y,
            style_attrs,
            html_escape(&self.content)
        )
    }
}

/// Escape HTML special characters
fn html_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&#39;")
}

/// Arrow marker for line endings
#[derive(Debug, Clone)]
pub struct ArrowMarker {
    /// Marker ID
    pub id: String,
    /// Arrow color
    pub color: Color,
    /// Arrow size
    pub size: f32,
}

impl ArrowMarker {
    /// Create a new arrow marker
    pub fn new(id: &str, color: Color) -> Self {
        Self {
            id: id.to_string(),
            color,
            size: 10.0,
        }
    }

    /// Set the arrow size
    pub fn with_size(mut self, size: f32) -> Self {
        self.size = size;
        self
    }

    /// Render to SVG marker definition
    pub fn to_svg_def(&self) -> String {
        format!(
            r#"<marker id="{}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="{}" markerHeight="{}" orient="auto-start-reverse">
  <path d="M 0 0 L 10 5 L 0 10 z" fill="{}"/>
</marker>"#,
            self.id,
            self.size,
            self.size,
            self.color.to_css_hex()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(3.0, 4.0);
        assert!((p1.distance(&p2) - 5.0).abs() < 0.001);
    }

    #[test]
    fn test_point_midpoint() {
        let p1 = Point::new(0.0, 0.0);
        let p2 = Point::new(10.0, 10.0);
        let mid = p1.midpoint(&p2);
        assert_eq!(mid.x, 5.0);
        assert_eq!(mid.y, 5.0);
    }

    #[test]
    fn test_rect_creation() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.position.x, 10.0);
        assert_eq!(rect.position.y, 20.0);
        assert_eq!(rect.size.width, 100.0);
        assert_eq!(rect.size.height, 50.0);
    }

    #[test]
    fn test_rect_center() {
        let rect = Rect::new(0.0, 0.0, 100.0, 50.0);
        let center = rect.center();
        assert_eq!(center.x, 50.0);
        assert_eq!(center.y, 25.0);
    }

    #[test]
    fn test_rect_contains() {
        let rect = Rect::new(0.0, 0.0, 100.0, 100.0);
        assert!(rect.contains(&Point::new(50.0, 50.0)));
        assert!(!rect.contains(&Point::new(150.0, 50.0)));
    }

    #[test]
    fn test_rect_intersects() {
        let r1 = Rect::new(0.0, 0.0, 100.0, 100.0);
        let r2 = Rect::new(50.0, 50.0, 100.0, 100.0);
        let r3 = Rect::new(200.0, 200.0, 50.0, 50.0);

        assert!(r1.intersects(&r2));
        assert!(!r1.intersects(&r3));
    }

    #[test]
    fn test_rect_to_svg() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0)
            .with_fill(Color::rgb(255, 0, 0))
            .with_stroke(Color::rgb(0, 0, 0), 2.0);

        let svg = rect.to_svg();
        assert!(svg.contains("x=\"10\""));
        assert!(svg.contains("fill=\"#FF0000\""));
        assert!(svg.contains("stroke=\"#000000\""));
    }

    #[test]
    fn test_circle_creation() {
        let circle = Circle::new(50.0, 50.0, 25.0);
        assert_eq!(circle.center.x, 50.0);
        assert_eq!(circle.radius, 25.0);
    }

    #[test]
    fn test_circle_contains() {
        let circle = Circle::new(50.0, 50.0, 25.0);
        assert!(circle.contains(&Point::new(50.0, 50.0)));
        assert!(circle.contains(&Point::new(60.0, 50.0)));
        assert!(!circle.contains(&Point::new(100.0, 100.0)));
    }

    #[test]
    fn test_circle_to_svg() {
        let circle = Circle::new(50.0, 50.0, 25.0).with_fill(Color::rgb(0, 255, 0));

        let svg = circle.to_svg();
        assert!(svg.contains("cx=\"50\""));
        assert!(svg.contains("r=\"25\""));
        assert!(svg.contains("fill=\"#00FF00\""));
    }

    #[test]
    fn test_line_creation() {
        let line = Line::new(0.0, 0.0, 100.0, 100.0);
        assert!((line.length() - 141.42).abs() < 0.1);
    }

    #[test]
    fn test_path_builder() {
        let path = Path::new()
            .move_to(0.0, 0.0)
            .line_to(100.0, 0.0)
            .line_to(100.0, 100.0)
            .close();

        let data = path.to_path_data();
        assert!(data.contains("M 0 0"));
        assert!(data.contains("L 100 0"));
        assert!(data.ends_with("Z"));
    }

    #[test]
    fn test_text_creation() {
        let text = Text::new(10.0, 20.0, "Hello");
        assert_eq!(text.content, "Hello");
        assert_eq!(text.position.x, 10.0);
    }

    #[test]
    fn test_text_escaping() {
        let text = Text::new(0.0, 0.0, "<script>");
        let svg = text.to_svg();
        assert!(svg.contains("&lt;script&gt;"));
    }

    #[test]
    fn test_arrow_marker() {
        let arrow = ArrowMarker::new("arrow1", Color::rgb(0, 0, 0));
        let svg = arrow.to_svg_def();
        assert!(svg.contains("id=\"arrow1\""));
        assert!(svg.contains("fill=\"#000000\""));
    }

    #[test]
    fn test_point_new() {
        let p = Point::new(5.0, 10.0);
        assert_eq!(p.x, 5.0);
        assert_eq!(p.y, 10.0);
    }

    #[test]
    fn test_point_default() {
        let p = Point::default();
        assert_eq!(p.x, 0.0);
        assert_eq!(p.y, 0.0);
    }

    #[test]
    fn test_size_new() {
        let s = Size::new(100.0, 50.0);
        assert_eq!(s.width, 100.0);
        assert_eq!(s.height, 50.0);
    }

    #[test]
    fn test_size_area() {
        let s = Size::new(10.0, 5.0);
        assert_eq!(s.area(), 50.0);
    }

    #[test]
    fn test_size_default() {
        let s = Size::default();
        assert_eq!(s.width, 0.0);
        assert_eq!(s.height, 0.0);
    }

    #[test]
    fn test_rect_right_bottom() {
        let rect = Rect::new(10.0, 20.0, 100.0, 50.0);
        assert_eq!(rect.right(), 110.0);
        assert_eq!(rect.bottom(), 70.0);
    }

    #[test]
    fn test_rect_default() {
        let rect = Rect::default();
        assert_eq!(rect.position.x, 0.0);
        assert_eq!(rect.size.width, 100.0);
    }

    #[test]
    fn test_rect_with_radius() {
        let rect = Rect::new(0.0, 0.0, 50.0, 50.0).with_radius(8.0);
        assert_eq!(rect.corner_radius, 8.0);
        let svg = rect.to_svg();
        assert!(svg.contains("rx=\"8\""));
    }

    #[test]
    fn test_circle_bounds() {
        let circle = Circle::new(50.0, 50.0, 25.0);
        let bounds = circle.bounds();
        assert_eq!(bounds.position.x, 25.0);
        assert_eq!(bounds.position.y, 25.0);
        assert_eq!(bounds.size.width, 50.0);
        assert_eq!(bounds.size.height, 50.0);
    }

    #[test]
    fn test_circle_intersects() {
        let c1 = Circle::new(0.0, 0.0, 10.0);
        let c2 = Circle::new(15.0, 0.0, 10.0);
        let c3 = Circle::new(50.0, 0.0, 10.0);
        assert!(c1.intersects(&c2));  // Overlapping
        assert!(!c1.intersects(&c3)); // Not touching
    }

    #[test]
    fn test_circle_with_stroke() {
        let circle = Circle::new(50.0, 50.0, 25.0)
            .with_stroke(Color::rgb(0, 0, 255), 3.0);
        let svg = circle.to_svg();
        assert!(svg.contains("stroke=\"#0000FF\""));
        assert!(svg.contains("stroke-width=\"3\""));
    }

    #[test]
    fn test_circle_default() {
        let circle = Circle::default();
        assert_eq!(circle.center.x, 50.0);
        assert_eq!(circle.radius, 25.0);
    }

    #[test]
    fn test_line_midpoint() {
        let line = Line::new(0.0, 0.0, 100.0, 100.0);
        let mid = line.midpoint();
        assert_eq!(mid.x, 50.0);
        assert_eq!(mid.y, 50.0);
    }

    #[test]
    fn test_line_with_stroke() {
        let line = Line::new(0.0, 0.0, 100.0, 0.0)
            .with_stroke(Color::rgb(255, 0, 0));
        assert_eq!(line.stroke, Color::rgb(255, 0, 0));
    }

    #[test]
    fn test_line_with_stroke_width() {
        let line = Line::new(0.0, 0.0, 100.0, 0.0)
            .with_stroke_width(5.0);
        assert_eq!(line.stroke_width, 5.0);
    }

    #[test]
    fn test_line_with_dash() {
        let line = Line::new(0.0, 0.0, 100.0, 0.0)
            .with_dash("5,5");
        let svg = line.to_svg();
        assert!(svg.contains("stroke-dasharray=\"5,5\""));
    }

    #[test]
    fn test_line_to_svg() {
        let line = Line::new(10.0, 20.0, 30.0, 40.0);
        let svg = line.to_svg();
        assert!(svg.contains("x1=\"10\""));
        assert!(svg.contains("y1=\"20\""));
        assert!(svg.contains("x2=\"30\""));
        assert!(svg.contains("y2=\"40\""));
    }

    #[test]
    fn test_path_quad_curve() {
        let path = Path::new()
            .move_to(0.0, 0.0)
            .quad_to(50.0, 50.0, 100.0, 0.0);
        let data = path.to_path_data();
        assert!(data.contains("Q 50 50"));
    }

    #[test]
    fn test_path_cubic_curve() {
        let path = Path::new()
            .move_to(0.0, 0.0)
            .cubic_to(25.0, 50.0, 75.0, 50.0, 100.0, 0.0);
        let data = path.to_path_data();
        assert!(data.contains("C 25 50"));
    }

    #[test]
    fn test_path_with_fill_stroke() {
        let path = Path::new()
            .move_to(0.0, 0.0)
            .line_to(100.0, 100.0)
            .with_fill(Color::rgb(255, 0, 0))
            .with_stroke(Color::rgb(0, 0, 0), 2.0);
        let svg = path.to_svg();
        assert!(svg.contains("fill=\"#FF0000\""));
        assert!(svg.contains("stroke=\"#000000\""));
    }

    #[test]
    fn test_path_default() {
        let path = Path::default();
        assert!(path.commands.is_empty());
    }

    #[test]
    fn test_text_with_style() {
        use crate::oracle::svg::typography::{TextStyle, FontWeight};
        let style = TextStyle::new(16.0, FontWeight::Bold);
        let text = Text::new(0.0, 0.0, "Hello").with_style(style);
        let svg = text.to_svg();
        assert!(svg.contains("font-weight=\"700\""));
    }

    #[test]
    fn test_arrow_marker_with_size() {
        let arrow = ArrowMarker::new("a1", Color::rgb(0, 0, 0))
            .with_size(20.0);
        assert_eq!(arrow.size, 20.0);
        let svg = arrow.to_svg_def();
        assert!(svg.contains("markerWidth=\"20\""));
    }

    #[test]
    fn test_html_escape() {
        let text = Text::new(0.0, 0.0, "a & b < c > d \"e\" 'f'");
        let svg = text.to_svg();
        assert!(svg.contains("&amp;"));
        assert!(svg.contains("&lt;"));
        assert!(svg.contains("&gt;"));
        assert!(svg.contains("&quot;"));
        assert!(svg.contains("&#39;"));
    }

    #[test]
    fn test_rect_no_fill() {
        let rect = Rect::new(0.0, 0.0, 50.0, 50.0);
        let svg = rect.to_svg();
        assert!(svg.contains("fill=\"none\""));
    }

    #[test]
    fn test_circle_no_fill() {
        let circle = Circle::new(50.0, 50.0, 25.0);
        let svg = circle.to_svg();
        assert!(svg.contains("fill=\"none\""));
    }

    #[test]
    fn test_path_no_fill() {
        let path = Path::new().move_to(0.0, 0.0).line_to(100.0, 100.0);
        let svg = path.to_svg();
        assert!(svg.contains("fill=\"none\""));
    }
}
