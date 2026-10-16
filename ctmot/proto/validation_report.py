"""Human-readable summary and standalone plots for real-data prediction checks."""

import numpy as np


def write_report(report, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    shadow = report["prediction"]
    snapshot = shadow["snapshot"]
    acceleration = shadow["acceleration_estimate"]
    text = ["# 실제 history 기반 CT prediction 검증", "",
            "- 사례: {} / ID{} / anchor frame{} (0-based)".format(report["case"]["scene"], snapshot["track_id"], snapshot["anchor_frame"]),
            "- 공통 초기 [px,py,pz,vx,vy,vz]: `{}`".format(snapshot["state_pv"]),
            "- 실제 초기 timestamp: `{}` µs".format(snapshot["timestamp_us"]),
            "- 과거 관측 frame: `{}` (신규 생성 관측 제외)".format([x["frame_idx"] for x in snapshot["history"]]),
            "- CA 추정 상태: `{}` / 가속도: `{}` m/s²".format(acceleration["status"], acceleration["acceleration_mps2"]),
            "- 속도 회귀의 절편은 초기 속도를 교체하지 않음. Z는 모든 모델에서 유지.",
            "- GT는 prediction 저장 이후 평가에만 사용. 검출 대응은 같은 클래스 GT 중심에 가장 가까운 후보를 반경 내에서 선택.",
            "- 크기/yaw는 같은 query의 baseline prediction 값. CT는 중심만 교체.",
            "- gate 통과는 Hungarian 배정 성공이나 IDS 개선을 의미하지 않음.", "",
            "| Query frame | Model | Actual Δt (s) | GT XY error (m) | GT Z error (m) | GT XYZ error (m) | Cost | Spatial gate | Cost gate | Both gates |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for query, evaluation in zip(shadow["queries"], report["evaluation"]):
        models = [m for m in ["baseline", "ct_cv", "ct_ca"] if m in query["models"]]
        dt = (query["timestamp_us"] - snapshot["timestamp_us"]) / 1e6
        for name in models:
            metrics = evaluation["measurements"][name]; assoc = metrics["association"]
            text.append("| {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {} | {} | {} | {} |".format(
                query["frame_idx"], name, dt, metrics["gt_xy_error_m"], metrics["gt_z_error_m"], metrics["gt_xyz_error_m"],
                "{:.6f}".format(assoc["cost"]) if assoc else "N/A", *[
                    str(assoc[k]) if assoc else "N/A" for k in ["spatial_gate_pass", "cost_gate_pass", "gate_pass"]]))
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
        colors = {"baseline": "#ba3434", "ct_cv": "#287fbd", "ct_ca": "#d68c18"}
        origin = np.array(snapshot["state_pv"][:2])
        axes[0].scatter(0, 0, color="gray", marker="s", label="Shared posterior at anchor")
        gt = np.array(evaluation["gt"]["translation"][:2]) - origin
        axes[0].scatter(*gt, color="black", marker="*", s=130, label="Query GT")
        for name in models:
            position = np.array(query["models"][name]["position_m"][:2]) - origin
            axes[0].plot([0, position[0]], [0, position[1]], color=colors[name], alpha=.5)
            axes[0].scatter(*position, color=colors[name], label=name)
        axes[0].set_aspect("equal"); axes[0].set_xlabel("Global X offset from posterior (m)")
        axes[0].set_ylabel("Global Y offset from posterior (m)"); axes[0].legend(fontsize=8)
        axes[0].set_title("Same posterior, different predictions")
        values = [evaluation["measurements"][m]["gt_xy_error_m"] for m in models]
        axes[1].bar(models, values, color=[colors[m] for m in models])
        for i, value in enumerate(values): axes[1].text(i, value + .025, "{:.4f}".format(value), ha="center")
        axes[1].set_ylim(0, max(values) * 1.2 + .05); axes[1].set_ylabel("GT XY error (m)")
        axes[1].set_title("Position error at actual query time")
        candidate = evaluation["candidate"]
        if candidate:
            costs = [evaluation["measurements"][m]["association"]["cost"] for m in models]
            axes[2].bar(models, costs, color=[colors[m] for m in models])
            for i, value in enumerate(costs): axes[2].text(i, value + .025, "{:.4f}".format(value), ha="center")
            axes[2].axhline(candidate["cost_limit"], color="black", ls="--", label="Original cost limit")
            axes[2].set_ylim(0, max(costs + [candidate["cost_limit"]]) * 1.2)
            axes[2].legend(fontsize=8)
        else:
            axes[2].text(.5, .5, "No supported detection candidate", ha="center", transform=axes[2].transAxes)
        axes[2].set_title("Original geometry cost; no Hungarian rerun")
        axes[2].set_ylabel("1 - GIoU3D")
        fig.suptitle("{} | ID{} | anchor {} -> key {} | actual dt={:.6f}s".format(
            report["case"]["scene"], snapshot["track_id"], snapshot["anchor_frame"], query["frame_idx"], dt))
        fig.savefig(output / ("comparison_frame{:06d}.png".format(query["frame_idx"])), dpi=150)
        plt.close(fig)
    text += ["", "## 재현 정보", "", "입력 파일 hash 및 코드 hash, 전체 prediction, history, GT, gate 결과는 `report.json`에 있습니다.",
             "`predictions.json`은 GT 평가 전에 저장된 prediction입니다. 원본 결과·trace·config의 실행 전후 hash 일치를 확인했습니다.",
             "이 검증은 keyframe에서의 단일 anchor 개방 루프 비교입니다. 전체 tracker의 ID와 원본 high 결과는 수정하지 않습니다."]
    (output / "report.md").write_text("\n".join(text) + "\n")
