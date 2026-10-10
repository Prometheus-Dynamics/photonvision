//! The pipeline configuration PhotonVision sends with `StyxJNI.configure` (JSON).

use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineConfig {
    /// `"tag36h11"` (default) or `"tag16h5"`.
    #[serde(default = "default_family")]
    pub family: String,
    /// Edge length of the tags, metres (the black border's outer edge).
    pub tag_size_m: f64,
    /// Camera intrinsics and lens; without them no pose is solved (corners and ids still are).
    pub calibration: Option<CalibrationConfig>,
    /// Solve one camera pose from every seen tag of `field_tags`.
    #[serde(default)]
    pub multi_tag: bool,
    /// Known tag poses in the field frame (WPILib: NWU, tag `x` out of the face).
    #[serde(default)]
    pub field_tags: Vec<FieldTag>,
    /// A full-frame search at least every this many frames (Eidos tracking).
    #[serde(default = "default_full_search")]
    pub full_search_every: u32,
    /// Also encode the processed stream (the grey frame with the detected tags outlined).
    #[serde(default)]
    pub processed_stream: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CalibrationConfig {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    #[serde(default)]
    pub skew: f64,
    #[serde(default)]
    pub k1: f64,
    #[serde(default)]
    pub k2: f64,
    #[serde(default)]
    pub k3: f64,
    #[serde(default)]
    pub k4: f64,
    #[serde(default)]
    pub k5: f64,
    #[serde(default)]
    pub k6: f64,
    #[serde(default)]
    pub p1: f64,
    #[serde(default)]
    pub p2: f64,
    /// Kannala-Brandt (OpenCV `cv::fisheye`) instead of Brown-Conrady.
    #[serde(default)]
    pub fisheye: bool,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldTag {
    pub id: u32,
    /// Overrides `tag_size_m` for this tag.
    pub size_m: Option<f64>,
    pub x: f64,
    pub y: f64,
    pub z: f64,
    pub qw: f64,
    pub qx: f64,
    pub qy: f64,
    pub qz: f64,
}

fn default_family() -> String {
    "tag36h11".into()
}

fn default_full_search() -> u32 {
    8
}

impl PipelineConfig {
    pub fn parse(json: &str) -> Result<Self, String> {
        let config: Self = serde_json::from_str(json).map_err(|e| format!("bad config: {e}"))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<(), String> {
        if !(self.tag_size_m.is_finite() && self.tag_size_m > 0.0) {
            return Err("tagSizeM must be a positive length".into());
        }
        match self.family.as_str() {
            "tag36h11" | "tag16h5" => {}
            other => return Err(format!("unknown tag family '{other}'")),
        }
        if let Some(c) = &self.calibration
            && !(c.fx > 0.0 && c.fy > 0.0 && c.fx.is_finite() && c.fy.is_finite())
        {
            return Err("calibration focal lengths must be positive".into());
        }
        for tag in &self.field_tags {
            if let Some(size) = tag.size_m
                && !(size.is_finite() && size > 0.0)
            {
                return Err(format!("field tag {} has a bad size", tag.id));
            }
        }
        Ok(())
    }

    pub fn tag_size_for(&self, id: u32) -> f64 {
        self.field_tags
            .iter()
            .find(|t| t.id == id)
            .and_then(|t| t.size_m)
            .unwrap_or(self.tag_size_m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_and_full_configs() {
        let minimal = PipelineConfig::parse(r#"{"tagSizeM":0.165}"#).unwrap();
        assert_eq!(minimal.family, "tag36h11");
        assert_eq!(minimal.full_search_every, 8);
        assert!(minimal.calibration.is_none());

        let full = PipelineConfig::parse(
            r#"{"family":"tag16h5","tagSizeM":0.15,"multiTag":true,
                "calibration":{"fx":800,"fy":800,"cx":640,"cy":400,"k1":0.01,"fisheye":true},
                "fieldTags":[{"id":3,"sizeM":0.2,"x":1,"y":2,"z":0.5,"qw":1,"qx":0,"qy":0,"qz":0}]}"#,
        )
        .unwrap();
        assert!(full.multi_tag);
        assert!(full.calibration.unwrap().fisheye);
        assert_eq!(full.tag_size_for(3), 0.2);
        assert_eq!(full.tag_size_for(4), 0.15);
    }

    #[test]
    fn rejects_bad_values() {
        assert!(PipelineConfig::parse(r#"{"tagSizeM":0}"#).is_err());
        assert!(PipelineConfig::parse(r#"{"tagSizeM":0.1,"family":"nope"}"#).is_err());
        assert!(PipelineConfig::parse("not json").is_err());
    }
}
