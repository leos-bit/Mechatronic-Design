# Current System Status

## Pick-A-Bot CV Pipeline

**Status:** Working end-to-end for detection, tracking, and position output

### Implemented
- YOLO runtime detection from main branch
- `bottle`, `can`, and optional `six_pack` support
- ByteTrack object tracking
- Refined centroid estimation inside each detection
- Homography-based world coordinate mapping
- Auto-labeling pipeline for 2-class and 3-class datasets

### Open Issues
- `six_pack` classification still partly heuristic on older 2-class models
- Centroid accuracy depends on lighting, contrast, and camera geometry
- Belt-level quantitative validation is not complete yet

### Next Steps
- Validate centroid and world-coordinate accuracy on the belt
- Test six-pack classification across more videos
- Build cleaner training data for a dedicated 3-class model
