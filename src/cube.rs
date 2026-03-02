//! ZOT Cube — 3D Substrate State Memory
//!
//! The organism's sensor readings [mem, clock, alloc] define a point in 3D space.
//! The cube discretizes this space into bins and accumulates experience:
//! which regions are self, which are threat, which receptors dominate where.
//!
//! A Rubik's cube has 6 faces, each a grid. Our cube has 6 faces too —
//! 3 axis-aligned cross-sections (mem×clock, mem×alloc, clock×alloc)
//! viewed from both directions (low and high on the third axis).
//!
//! "Solved" = homeostasis. All faces show self-consistent patterns.
//! "Scrambled" = threat. Faces show anomalous configurations.
//!
//! The cube provides:
//!   1. Spatial memory — which regions of substrate space are dangerous
//!   2. Advisory vote — a geometric decision independent of receptor cascade
//!   3. Receptor locality — which receptors own which regions
//!   4. Scramble detection — how many faces are disturbed simultaneously

/// Resolution per axis. 8×8×8 = 512 cells. Small enough to fill fast,
/// large enough to capture structure.
const BINS: usize = 8;

/// A single voxel in substrate space.
#[derive(Clone, Default)]
pub struct Voxel {
    pub visits: u32,
    pub threat_visits: u32,
    pub correct: u32,
    pub total_decided: u32,
    /// Index of the receptor that fired most often in this region
    pub dominant_receptor: Option<usize>,
    dominant_fires: u32,
}

impl Voxel {
    pub fn threat_ratio(&self) -> f64 {
        if self.visits == 0 { 0.0 } else { self.threat_visits as f64 / self.visits as f64 }
    }

    pub fn accuracy(&self) -> f64 {
        if self.total_decided == 0 { 0.5 } else { self.correct as f64 / self.total_decided as f64 }
    }
}

/// The 3D substrate state cube.
pub struct Cube {
    /// 8×8×8 voxel grid. Index: [mem_bin][clock_bin][alloc_bin]
    voxels: [[[Voxel; BINS]; BINS]; BINS],
    /// Per-axis bin edges, computed from calibration percentiles
    edges: [[f64; BINS + 1]; 3],
    /// Running count of how many faces are "disturbed" (recent threat ratio > threshold)
    pub scramble_level: u32,
}

impl Cube {
    /// Build the cube from calibration data. Bin edges are evenly spaced
    /// across a wide range to capture both self and threat readings.
    /// We extend far below calibration (threats push readings down)
    /// and moderately above (some sensors spike under load).
    pub fn from_calibration(sensor_stats: &[(f64, f64); 3]) -> Self {
        let mut edges = [[0.0f64; BINS + 1]; 3];
        for axis in 0..3 {
            let (lo, hi) = sensor_stats[axis];
            let range = hi - lo;
            // Threats push readings DOWN (memory drops, clock drops, alloc drops).
            // Extend 100% below calibration low to capture threat territory.
            // Extend 50% above calibration high for spikes.
            let ext_lo = lo - range * 1.0;
            let ext_hi = hi + range * 0.5;
            let step = (ext_hi - ext_lo) / BINS as f64;
            for b in 0..=BINS {
                edges[axis][b] = ext_lo + step * b as f64;
            }
        }
        Cube {
            voxels: Default::default(),
            edges,
            scramble_level: 0,
        }
    }

    /// Map a raw sensor value to a bin index (0..BINS-1), clamped.
    fn bin(&self, axis: usize, value: f64) -> usize {
        let edges = &self.edges[axis];
        for i in 0..BINS {
            if value < edges[i + 1] {
                return i;
            }
        }
        BINS - 1
    }

    /// Convert a sensor reading to cube coordinates.
    pub fn coords(&self, reading: &[f64; 3]) -> [usize; 3] {
        [
            self.bin(0, reading[0]),
            self.bin(1, reading[1]),
            self.bin(2, reading[2]),
        ]
    }

    /// Record an observation: the organism saw this reading, decided block/allow,
    /// and the ground truth was threat/quiet.
    pub fn observe(
        &mut self,
        reading: &[f64; 3],
        is_threat: bool,
        blocked: bool,
        firing_receptor: Option<usize>,
    ) {
        let [m, c, a] = self.coords(reading);
        let v = &mut self.voxels[m][c][a];
        v.visits += 1;
        if is_threat {
            v.threat_visits += 1;
        }
        let correct = blocked == is_threat;
        v.total_decided += 1;
        if correct {
            v.correct += 1;
        }
        if let Some(rid) = firing_receptor {
            // Track dominant receptor per voxel (simple majority)
            if v.dominant_receptor == Some(rid) {
                v.dominant_fires += 1;
            } else if v.dominant_fires == 0 {
                v.dominant_receptor = Some(rid);
                v.dominant_fires = 1;
            } else {
                // Decay toward new receptor
                v.dominant_fires -= 1;
                if v.dominant_fires == 0 {
                    v.dominant_receptor = Some(rid);
                    v.dominant_fires = 1;
                }
            }
        }
        self.update_scramble();
    }

    /// Advisory vote: based purely on spatial memory, should we block?
    /// Returns (should_block, confidence) where confidence is 0.0-1.0.
    /// This does NOT replace the receptor vote. It's a second opinion.
    pub fn advisory_vote(&self, reading: &[f64; 3]) -> (bool, f64) {
        let [m, c, a] = self.coords(reading);
        let v = &self.voxels[m][c][a];

        if v.visits < 3 {
            // Not enough data in this voxel — check neighbors
            let neighbor_threat = self.neighbor_threat_ratio(m, c, a);
            if neighbor_threat > 0.6 {
                return (true, neighbor_threat * 0.5); // low confidence from neighbors
            }
            return (false, 0.0); // abstain
        }

        let threat_r = v.threat_ratio();
        let confidence = (v.visits as f64 / 20.0).min(1.0); // confidence grows with visits
        (threat_r > 0.5, threat_r * confidence)
    }

    // ================================================================
    // THE ORGANISM FUNCTION: behavior = f(nature, environment)
    // ================================================================
    //
    // solve() IS the organism function materialized as geometry.
    //
    // The receptors sculpt the cube through Darwinian selection.
    // The cube accumulates their decisions as spatial memory.
    // solve() reads the shape and decides geometrically — O(1).
    //
    // Like an enzyme: the substrate state is a molecule.
    // The cube's threat surface is the binding site.
    // The molecule fits or it doesn't. Physics decides.
    // No computation. Geometry.
    //
    // Returns: SolveResult with the decision and all the geometry.

    /// The organism function. behavior = f(nature, environment).
    ///
    /// nature = the cube's learned shape (the manifold)
    /// environment = the current sensor reading (a point in substrate space)
    /// behavior = block or allow (the decision)
    ///
    /// Three layers of geometric reasoning:
    ///   1. VOXEL: direct lookup — has this exact region been seen before?
    ///   2. NEIGHBORHOOD: diffusion — what do adjacent regions say?
    ///   3. FACE: scramble — how many dimensions are disturbed?
    ///
    /// Each layer has a confidence. The organism trusts the most confident layer.
    /// When the cube is young (few visits), confidence is low → fall back to receptors.
    /// When the cube is mature, the shape decides. The receptors become sculptors only.
    pub fn solve(&self, reading: &[f64; 3]) -> SolveResult {
        let [m, c, a] = self.coords(reading);
        let v = &self.voxels[m][c][a];

        // --- LAYER 1: VOXEL (direct memory) ---
        let voxel_decision = if v.visits >= 5 {
            let tr = v.threat_ratio();
            // Confidence requires substantial evidence: 30+ visits for full confidence
            // AND the voxel must be clearly self or clearly threat (not 50/50)
            let clarity = (tr - 0.5).abs() * 2.0; // 0.0 at 50/50, 1.0 at 0% or 100%
            let visit_conf = (v.visits as f64 / 30.0).min(1.0);
            let conf = visit_conf * clarity * v.accuracy();
            Some((tr > 0.5, conf, tr))
        } else {
            None
        };

        // --- LAYER 2: NEIGHBORHOOD (spatial diffusion) ---
        // Like chemical signaling between adjacent cells.
        // A voxel with no experience asks its neighbors.
        let neighborhood = self.neighborhood_solve(m, c, a);

        // --- LAYER 3: FACE SCRAMBLE (dimensional analysis) ---
        // How many of the 6 faces are disturbed?
        // 0 faces = solved = self. 3+ faces = scrambled = threat.
        // This is the Rubik's insight: you don't check every sticker.
        // You check how many faces are wrong.
        let face_decision = if self.scramble_level >= 3 {
            Some((true, 0.6, self.scramble_level))
        } else if self.scramble_level == 0 {
            // All faces clean — but only trust this if we have enough data
            let total_visits: u64 = self.total_visits();
            if total_visits > 50 {
                Some((false, 0.4, 0))
            } else {
                None
            }
        } else {
            None // ambiguous scramble — don't trust face layer
        };

        // --- FUSION: pick the most confident layer ---
        let mut best_block = false;
        let mut best_conf = 0.0f64;
        let mut deciding_layer = Layer::None;

        if let Some((block, conf, _)) = voxel_decision {
            if conf > best_conf {
                best_block = block;
                best_conf = conf;
                deciding_layer = Layer::Voxel;
            }
        }
        if let Some((block, conf)) = neighborhood {
            if conf > best_conf {
                best_block = block;
                best_conf = conf;
                deciding_layer = Layer::Neighborhood;
            }
        }
        if let Some((block, conf, _)) = face_decision {
            if conf > best_conf {
                best_block = block;
                best_conf = conf;
                deciding_layer = Layer::Face;
            }
        }

        SolveResult {
            block: best_block,
            confidence: best_conf,
            layer: deciding_layer,
            voxel_threat_ratio: v.threat_ratio(),
            voxel_visits: v.visits,
            scramble: self.scramble_level,
            coords: [m, c, a],
        }
    }

    /// Neighborhood solve: weighted average of the 3×3×3 neighborhood.
    /// Closer neighbors (face-adjacent) weighted more than corner-adjacent.
    fn neighborhood_solve(&self, m: usize, c: usize, a: usize) -> Option<(bool, f64)> {
        let mut weighted_threat = 0.0;
        let mut total_weight = 0.0;
        let mut contributing = 0u32;

        for dm in -1i32..=1 {
            for dc in -1i32..=1 {
                for da in -1i32..=1 {
                    let nm = (m as i32 + dm).clamp(0, BINS as i32 - 1) as usize;
                    let nc = (c as i32 + dc).clamp(0, BINS as i32 - 1) as usize;
                    let na = (a as i32 + da).clamp(0, BINS as i32 - 1) as usize;
                    let nv = &self.voxels[nm][nc][na];
                    if nv.visits < 2 { continue; }

                    // Manhattan distance determines weight: face=1, edge=0.5, corner=0.25
                    let dist = dm.unsigned_abs() + dc.unsigned_abs() + da.unsigned_abs();
                    let w = match dist {
                        0 => 2.0, // self — strongest
                        1 => 1.0, // face-adjacent
                        2 => 0.5, // edge-adjacent
                        3 => 0.25, // corner-adjacent
                        _ => 0.0,
                    };
                    let visit_w = (nv.visits as f64 / 10.0).min(1.0);
                    let eff_w = w * visit_w;
                    weighted_threat += nv.threat_ratio() * eff_w;
                    total_weight += eff_w;
                    contributing += 1;
                }
            }
        }

        if contributing < 3 || total_weight < 1.0 {
            return None; // not enough spatial context
        }

        let avg_threat = weighted_threat / total_weight;
        let conf = (contributing as f64 / 15.0).min(1.0) * 0.7; // neighborhood is less confident than direct
        Some((avg_threat > 0.5, conf))
    }

    /// Total visits across all voxels.
    fn total_visits(&self) -> u64 {
        let mut total = 0u64;
        for m in 0..BINS {
            for c in 0..BINS {
                for a in 0..BINS {
                    total += self.voxels[m][c][a].visits as u64;
                }
            }
        }
        total
    }

    /// Average threat ratio of the 26 neighbors (3×3×3 minus center).
    fn neighbor_threat_ratio(&self, m: usize, c: usize, a: usize) -> f64 {
        let mut sum = 0.0;
        let mut count = 0;
        for dm in -1i32..=1 {
            for dc in -1i32..=1 {
                for da in -1i32..=1 {
                    if dm == 0 && dc == 0 && da == 0 { continue; }
                    let nm = (m as i32 + dm).clamp(0, BINS as i32 - 1) as usize;
                    let nc = (c as i32 + dc).clamp(0, BINS as i32 - 1) as usize;
                    let na = (a as i32 + da).clamp(0, BINS as i32 - 1) as usize;
                    let nv = &self.voxels[nm][nc][na];
                    if nv.visits > 0 {
                        sum += nv.threat_ratio();
                        count += 1;
                    }
                }
            }
        }
        if count == 0 { 0.0 } else { sum / count as f64 }
    }

    /// Count how many of the 6 faces are "scrambled" (have recent threat activity).
    /// A face is a 2D slice: fix one axis at its low or high end.
    fn update_scramble(&mut self) {
        let mut disturbed = 0u32;
        // 6 faces: axis 0 low/high, axis 1 low/high, axis 2 low/high
        let face_slices: [(usize, usize); 6] = [
            (0, 0), (0, BINS - 1),       // mem low, mem high
            (1, 0), (1, BINS - 1),       // clock low, clock high
            (2, 0), (2, BINS - 1),       // alloc low, alloc high
        ];
        for &(axis, fixed) in &face_slices {
            let mut face_threats = 0u32;
            let mut face_visits = 0u32;
            for i in 0..BINS {
                for j in 0..BINS {
                    let v = match axis {
                        0 => &self.voxels[fixed][i][j],
                        1 => &self.voxels[i][fixed][j],
                        _ => &self.voxels[i][j][fixed],
                    };
                    face_threats += v.threat_visits;
                    face_visits += v.visits;
                }
            }
            if face_visits > 0 && (face_threats as f64 / face_visits as f64) > 0.3 {
                disturbed += 1;
            }
        }
        self.scramble_level = disturbed;
    }

    /// Dump cube state for analysis. Returns a compact summary.
    pub fn summary(&self) -> CubeSummary {
        let mut total_visits = 0u64;
        let mut occupied = 0u32;
        let mut threat_voxels = 0u32;
        let mut self_voxels = 0u32;
        let mut contested = 0u32;
        let mut total_accuracy = 0.0f64;
        let mut accuracy_count = 0u32;

        for m in 0..BINS {
            for c in 0..BINS {
                for a in 0..BINS {
                    let v = &self.voxels[m][c][a];
                    if v.visits > 0 {
                        total_visits += v.visits as u64;
                        occupied += 1;
                        let tr = v.threat_ratio();
                        if tr > 0.7 { threat_voxels += 1; }
                        else if tr < 0.3 { self_voxels += 1; }
                        else { contested += 1; }
                        if v.total_decided > 0 {
                            total_accuracy += v.accuracy();
                            accuracy_count += 1;
                        }
                    }
                }
            }
        }

        CubeSummary {
            total_voxels: (BINS * BINS * BINS) as u32,
            occupied,
            threat_voxels,
            self_voxels,
            contested,
            total_visits,
            mean_accuracy: if accuracy_count > 0 { total_accuracy / accuracy_count as f64 } else { 0.0 },
            scramble_level: self.scramble_level,
        }
    }

    /// Write cube state to a file for the analyzer.
    pub fn write_cube_csv(&self, path: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;
        let mut f = File::create(path)?;
        writeln!(f, "mem_bin,clock_bin,alloc_bin,visits,threat_visits,correct,total_decided,threat_ratio,accuracy,dominant_receptor")?;
        for m in 0..BINS {
            for c in 0..BINS {
                for a in 0..BINS {
                    let v = &self.voxels[m][c][a];
                    if v.visits > 0 {
                        writeln!(
                            f,
                            "{m},{c},{a},{},{},{},{},{:.4},{:.4},{}",
                            v.visits, v.threat_visits, v.correct, v.total_decided,
                            v.threat_ratio(), v.accuracy(),
                            v.dominant_receptor.map(|r| r.to_string()).unwrap_or_else(|| "-".into()),
                        )?;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Which geometric layer made the decision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Layer {
    None,         // cube abstained — not enough data
    Voxel,        // direct memory at this exact point
    Neighborhood, // spatial diffusion from adjacent voxels
    Face,         // dimensional scramble analysis
}

impl std::fmt::Display for Layer {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Layer::None => write!(f, "---"),
            Layer::Voxel => write!(f, "VOX"),
            Layer::Neighborhood => write!(f, "NBR"),
            Layer::Face => write!(f, "FAC"),
        }
    }
}

/// The result of the organism function: behavior = f(nature, environment).
pub struct SolveResult {
    pub block: bool,
    pub confidence: f64,
    pub layer: Layer,
    pub voxel_threat_ratio: f64,
    pub voxel_visits: u32,
    pub scramble: u32,
    pub coords: [usize; 3],
}

impl SolveResult {
    /// Is the cube confident enough to override the receptor vote?
    /// The cube earns authority through experience. Below this threshold,
    /// the receptors decide and the cube learns.
    pub fn is_authoritative(&self) -> bool {
        // Need high confidence AND the voxel layer must be deciding
        // (not just neighborhood or face guessing)
        self.confidence >= 0.65 && self.layer == Layer::Voxel
    }
}

impl std::fmt::Display for SolveResult {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "{}@[{},{},{}] conf={:.2} tr={:.2} scr={} {}",
            self.layer,
            self.coords[0], self.coords[1], self.coords[2],
            self.confidence,
            self.voxel_threat_ratio,
            self.scramble,
            if self.block { "BLOCK" } else { "ALLOW" },
        )
    }
}

/// Compact summary for logging.
pub struct CubeSummary {
    pub total_voxels: u32,
    pub occupied: u32,
    pub threat_voxels: u32,
    pub self_voxels: u32,
    pub contested: u32,
    pub total_visits: u64,
    pub mean_accuracy: f64,
    pub scramble_level: u32,
}

impl std::fmt::Display for CubeSummary {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "cube: {}/{} voxels occupied (self={} threat={} contested={}) scramble={}/6 acc={:.1}%",
            self.occupied, self.total_voxels,
            self.self_voxels, self.threat_voxels, self.contested,
            self.scramble_level, self.mean_accuracy * 100.0,
        )
    }
}
