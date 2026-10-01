# P1-S1072-C-102 prediction rendering

All frames are 0-based. MID before19 is between key18–19; MID before20 is between key19–20. The panels show ID102 before association, not the corrected/output track or newborn ID124.

Red: exact saved prediction projected into the camera using its own sample_data pose and calibration. Green: actual true 2D detection. No extra score cutoff. Cyan omitted because no distinct best competing mid detection exists.

Files ending prediction_vs_3d.jpg show the correct GAAM-aligned 3D detection projected in green. These are the relevant geometric comparisons for the keyframe 3D gate.

prediction_2d_contact.jpg: rows = frame19 FRONT, frame20 FRONT, frame20 FRONT_RIGHT; columns = preceding MID, KEY.

Validation: six prediction projections and three aligned-detection projections match independent global→ego→camera pinhole calculations; all three mid predictions exactly match recorded association projections. Source results hash matches verified high replay.

| Frame | Camera | Camera timestamp µs | Mid 2D cost | Key 3D cost |
|---|---|---|---|---|
| mid19 | CAM_FRONT | 1542799711662460 | 0.7343922491303791 | N/A |
| key19 | CAM_FRONT | 1542799711862460 | None | 0.7521132020645214 |
| mid20 | CAM_FRONT | 1542799712162460 | 0.6876854555162143 | N/A |
| mid20 | CAM_FRONT_RIGHT | 1542799712170339 | 0.8235824412271618 | N/A |
| key20 | CAM_FRONT | 1542799712412460 | None | 1.1473571838075898 |
| key20 | CAM_FRONT_RIGHT | 1542799712420339 | None | 1.1473571838075898 |