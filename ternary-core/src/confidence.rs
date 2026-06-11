/// POLLN Three-Zone Confidence Cascade types.
///
/// Implements the Green/Yellow/Red tiered confidence model for
/// token-level routing decisions in a transformer.

use std::fmt;

/// Confidence zone for a single classification decision.
///
/// Each variant carries a score in [0.0, 1.0]:
/// - **Green**: High confidence (≥0.9) — route to full precision
/// - **Yellow**: Medium confidence (0.5–0.9) — route to ternary
/// - **Red**: Low confidence (<0.5) — re-sample or fallback
#[derive(Debug, Clone, PartialEq)]
pub enum Zone {
    Green(f64),
    Yellow(f64),
    Red(f64),
}

impl Zone {
    /// Returns the raw confidence score.
    pub fn score(&self) -> f64 {
        match self {
            Zone::Green(s) | Zone::Yellow(s) | Zone::Red(s) => *s,
        }
    }

    /// Returns the zone label as a string.
    pub fn label(&self) -> &str {
        match self {
            Zone::Green(_) => "green",
            Zone::Yellow(_) => "yellow",
            Zone::Red(_) => "red",
        }
    }

    /// Create a Zone from a raw score, applying the decision boundaries.
    pub fn from_score(score: f64) -> Self {
        let clamped = score.clamp(0.0, 1.0);
        if clamped >= 0.9 {
            Zone::Green(clamped)
        } else if clamped >= 0.5 {
            Zone::Yellow(clamped)
        } else {
            Zone::Red(clamped)
        }
    }
}

impl fmt::Display for Zone {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.3})", self.label(), self.score())
    }
}

/// Per-tile (per-token or per-patch) confidence tracking.
///
/// Tiles carry a score, a zone decision, and an optional trace
/// that records the decision path (e.g., which cascade stage triggered).
#[derive(Debug, Clone)]
pub struct TileConfidence {
    pub score: f64,
    pub zone: Zone,
    pub trace: Vec<String>,
}

impl TileConfidence {
    /// Create a new TileConfidence from a raw score.
    pub fn new(score: f64) -> Self {
        let zone = Zone::from_score(score);
        let trace = vec![format!("initial: {}", zone)];
        TileConfidence { score, zone, trace }
    }

    /// Manually set the zone (e.g., from an override).
    pub fn with_zone(mut self, zone: Zone) -> Self {
        self.trace
            .push(format!("zone_override: {}", zone.label()));
        self.zone = zone;
        self
    }

    /// Add a trace step.
    pub fn trace(mut self, msg: &str) -> Self {
        self.trace.push(msg.to_string());
        self
    }

    /// Returns the decision weight for routing: Green = 1.0, Yellow = 0.5, Red = 0.0.
    pub fn routing_weight(&self) -> f64 {
        match self.zone {
            Zone::Green(_) => 1.0,
            Zone::Yellow(_) => 0.5,
            Zone::Red(_) => 0.0,
        }
    }
}

/// Classify a slice of scores into zones.
pub fn confidence_cascade(scores: &[f64]) -> Vec<(f64, String)> {
    scores
        .iter()
        .map(|&s| {
            let zone = Zone::from_score(s);
            (zone.score(), zone.label().to_string())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_green_zone() {
        let z = Zone::from_score(0.95);
        assert!(matches!(z, Zone::Green(_)));
        assert!((z.score() - 0.95).abs() < 1e-10);
        assert_eq!(z.label(), "green");
    }

    #[test]
    fn test_yellow_zone() {
        let z = Zone::from_score(0.7);
        assert!(matches!(z, Zone::Yellow(_)));
        assert_eq!(z.label(), "yellow");
    }

    #[test]
    fn test_red_zone() {
        let z = Zone::from_score(0.3);
        assert!(matches!(z, Zone::Red(_)));
        assert_eq!(z.label(), "red");
    }

    #[test]
    fn test_boundary_green() {
        let z = Zone::from_score(0.9);
        assert!(matches!(z, Zone::Green(_)));
    }

    #[test]
    fn test_boundary_yellow_upper() {
        let z = Zone::from_score(0.8999);
        assert!(matches!(z, Zone::Yellow(_)));
    }

    #[test]
    fn test_boundary_yellow_lower() {
        let z = Zone::from_score(0.5);
        assert!(matches!(z, Zone::Yellow(_)));
    }

    #[test]
    fn test_boundary_red() {
        let z = Zone::from_score(0.4999);
        assert!(matches!(z, Zone::Red(_)));
    }

    #[test]
    fn test_clamp_overflow() {
        let z = Zone::from_score(1.5);
        assert!(matches!(z, Zone::Green(_)));
        assert!((z.score() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_clamp_underflow() {
        let z = Zone::from_score(-0.5);
        assert!(matches!(z, Zone::Red(_)));
        assert!((z.score() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_tile_confidence() {
        let tc = TileConfidence::new(0.85);
        assert!((tc.score - 0.85).abs() < 1e-10);
        assert!(matches!(tc.zone, Zone::Yellow(_)));
        assert!(!tc.trace.is_empty());
        assert!((tc.routing_weight() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cascade_all_zones() {
        let scores = vec![0.95, 0.65, 0.2, 0.9];
        let results = confidence_cascade(&scores);
        assert_eq!(results.len(), 4);
        assert_eq!(results[0].1, "green");
        assert_eq!(results[1].1, "yellow");
        assert_eq!(results[2].1, "red");
        assert_eq!(results[3].1, "green");
    }
}
