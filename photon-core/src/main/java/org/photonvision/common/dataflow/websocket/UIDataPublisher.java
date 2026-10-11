/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.common.dataflow.websocket;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.photonvision.common.dataflow.CVPipelineResultConsumer;
import org.photonvision.common.dataflow.DataChangeService;
import org.photonvision.common.dataflow.events.OutgoingUIEvent;
import org.photonvision.common.logging.LogGroup;
import org.photonvision.common.logging.Logger;
import org.photonvision.common.util.SerializationUtils;
import org.photonvision.styx.StyxJNI;
import org.photonvision.styx.StyxResult;
import org.photonvision.targeting.PhotonTrackedTarget;
import org.photonvision.vision.pipeline.result.CVPipelineResult;
import org.photonvision.vision.pipeline.result.CalibrationPipelineResult;
import org.photonvision.vision.pipeline.result.FocusPipelineResult;

public class UIDataPublisher implements CVPipelineResultConsumer {
    private static final Logger logger = new Logger(UIDataPublisher.class, LogGroup.VisionModule);

    private final String uniqueName;
    private long lastUIResultUpdateTime = 0;

    public UIDataPublisher(String uniqueName) {
        this.uniqueName = uniqueName;
    }

    @Override
    public void accept(CVPipelineResult result) {
        long now = System.currentTimeMillis();

        // only update the UI at 10hz
        if (lastUIResultUpdateTime + 1000.0 / 10.0 > now) return;

        Map<String, Object> dataMap = new HashMap<>();
        dataMap.put("sequenceID", result.sequenceID);
        dataMap.put("fps", result.fps);
        dataMap.put("latency", result.getLatencyMillis());
        List<Map<String, Object>> uiTargets = new ArrayList<>(result.targets.size());

        // We don't actually need to send targets during calibration and it can take up a lot (up to
        // 1.2Mbps for 60 snapshots) of target results with no pitch/yaw/etc set
        if (!(result instanceof CalibrationPipelineResult)) {
            if (result.photonTargets != null) {
                // The Styx backend's targets (no TrackedTarget behind them).
                for (var t : result.photonTargets) {
                    uiTargets.add(photonTargetToHashMap(t));
                }
            } else {
                for (var t : result.targets) {
                    uiTargets.add(t.toHashMap());
                }
            }
        }
        dataMap.put("processingMillis", result.getProcessingMillis());
        if (result.styxResult != null) {
            // Eidos's own time for the frame (detection, poses), in the camera session.
            dataMap.put("detectMillis", result.styxResult.processingNanos() / 1e6);
            long[] stages = result.styxResult.stages();
            // NetworkTables has published this result (its consumer runs before this one).
            stages[StyxResult.STAGE_PUBLISHED] = StyxJNI.nowBoottimeNanos();
            dataMap.put("stages", stageMillis(stages));
        }

        dataMap.put("targets", uiTargets);
        dataMap.put("classNames", result.objectDetectionClassNames);

        // Only send Multitag Results if they are present, similar to 3d pose
        if (result.multiTagResult.isPresent()) {
            var multitagData = new HashMap<String, Object>();
            multitagData.put(
                    "bestTransform",
                    SerializationUtils.transformToHashMap(result.multiTagResult.get().estimatedPose.best));
            multitagData.put(
                    "bestReprojectionError", result.multiTagResult.get().estimatedPose.bestReprojErr);
            multitagData.put("fiducialIDsUsed", result.multiTagResult.get().fiducialIDsUsed);
            dataMap.put("multitagResult", multitagData);
        }

        Map<String, Map<String, Object>> uiMap = new HashMap<>();
        uiMap.put(uniqueName, dataMap);

        if (result instanceof FocusPipelineResult focusResult) {
            dataMap.put("focus", focusResult.focus);
        }

        DataChangeService.getInstance()
                .publishEvent(OutgoingUIEvent.wrappedOf("updatePipelineResult", uiMap));
        lastUIResultUpdateTime = now;
    }

    private static final String[] STAGE_NAMES = {
        "sensor", "dequeued", "ispDone", "queued", "taken", "detectStart", "detectEnd", "packed",
        "javaReceived", "published"
    };

    /** Each known stage of a Styx frame, in ms after the sensor's start of frame. */
    static Map<String, Object> stageMillis(long[] stages) {
        var out = new HashMap<String, Object>();
        long origin = stages[StyxResult.STAGE_SENSOR];
        if (origin == 0) return out;
        for (int i = 1; i < stages.length && i < STAGE_NAMES.length; i++) {
            if (stages[i] != 0) out.put(STAGE_NAMES[i], (stages[i] - origin) / 1e6);
        }
        return out;
    }

    /** The keys {@code TrackedTarget.toHashMap()} sends, from a PhotonTrackedTarget. */
    static HashMap<String, Object> photonTargetToHashMap(PhotonTrackedTarget t) {
        var ret = new HashMap<String, Object>();
        ret.put("pitch", t.getPitch());
        ret.put("yaw", t.getYaw());
        ret.put("skew", t.getSkew());
        ret.put("area", t.getArea());
        ret.put("ambiguity", t.getPoseAmbiguity());
        ret.put("confidence", t.getDetectedObjectConfidence());
        ret.put("classId", t.getDetectedObjectClassID());
        if (t.getBestCameraToTarget() != null) {
            ret.put("pose", SerializationUtils.transformToHashMap(t.getBestCameraToTarget()));
        }
        ret.put("fiducialId", t.getFiducialId());
        return ret;
    }
}
