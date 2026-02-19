#include "tracker.h"
#include <algorithm>
#include <cmath>
#include <iostream>

BoTSORT::BoTSORT(float track_high_thresh, float track_low_thresh, 
                 float new_track_thresh, int track_buffer, float match_thresh)
    : frame_id(0), max_time_lost(track_buffer), track_high_thresh(track_high_thresh), track_low_thresh(track_low_thresh),
      new_track_thresh(new_track_thresh), match_thresh(match_thresh),
       next_id(1) {}

float BoTSORT::iou(const float* box1, const float* box2) {
    float x1 = std::max(box1[0], box2[0]);
    float y1 = std::max(box1[1], box2[1]);
    float x2 = std::min(box1[0] + box1[2], box2[0] + box2[2]);
    float y2 = std::min(box1[1] + box1[3], box2[1] + box2[3]);
    
    float w = std::max(0.0f, x2 - x1);
    float h = std::max(0.0f, y2 - y1);
    float inter_area = w * h;
    
    float area1 = box1[2] * box1[3];
    float area2 = box2[2] * box2[3];
    
    return inter_area / (area1 + area2 - inter_area + 1e-6);
}

std::vector<Track> BoTSORT::update(const std::vector<std::vector<float>>& input_dets) {
    frame_id++;
    
    std::vector<Track> activated_stracks;
    std::vector<Track> refind_stracks;
    std::vector<Track> lost_stracks;
    std::vector<Track> removed_stracks;
    
    // 1. Filter detections by score
    // det format: [class_id, score, x, y, w, h]
    // Simplified: Just use simple greedy IoU matching for this "stub" implementation update
    // A full SORT implementation is ~500 lines. We will implement a simplified Greedy Matcher.
    
    std::vector<Track> detections;
    for (const auto& det : input_dets) {
        if (det[1] >= track_low_thresh) {
            Track t;
            t.class_id = (int)det[0];
            t.score = det[1];
            t.bbox[0] = det[2]; t.bbox[1] = det[3]; t.bbox[2] = det[4]; t.bbox[3] = det[5];
            t.time_since_update = 0;
            detections.push_back(t);
        }
    }
    
    // 2. Match with existing tracked tracks
    std::vector<bool> det_matched(detections.size(), false);
    std::vector<bool> track_matched(tracked_stracks.size(), false);
    
    // DEBUG: Log track matching
    static int debug_frame = 0;
    if (debug_frame++ < 50) {
        std::cerr << "[TRACKER] Frame=" << frame_id << " existing_tracks=" << tracked_stracks.size() 
                  << " new_dets=" << detections.size() << std::endl;
    }
    
    for (size_t t = 0; t < tracked_stracks.size(); ++t) {
        float best_iou = 0.0f;
        int best_det_idx = -1;
        
        for (size_t d = 0; d < detections.size(); ++d) {
            if (det_matched[d]) continue;
            
            float overlap = iou(tracked_stracks[t].bbox, detections[d].bbox);
            if (overlap > best_iou) {
                best_iou = overlap;
                best_det_idx = d; // Fixed: was best_det_idx = overlap (wrong type)
            }
        }
        
        if (best_det_idx != -1 && best_iou > (1.0f - match_thresh)) {
             // Match found
             det_matched[best_det_idx] = true;
             track_matched[t] = true;
             if (debug_frame < 50) {
                 std::cerr << "[TRACKER] MATCHED track=" << tracked_stracks[t].track_id 
                           << " iou=" << best_iou << std::endl;
             }
             
             // Update track
             tracked_stracks[t].bbox[0] = detections[best_det_idx].bbox[0];
             tracked_stracks[t].bbox[1] = detections[best_det_idx].bbox[1];
             tracked_stracks[t].bbox[2] = detections[best_det_idx].bbox[2];
             tracked_stracks[t].bbox[3] = detections[best_det_idx].bbox[3];
             tracked_stracks[t].score = detections[best_det_idx].score;
             tracked_stracks[t].time_since_update = 0;
             tracked_stracks[t].last_seen = std::chrono::system_clock::now();
        } else {
             tracked_stracks[t].time_since_update++;
        }
    }
    
    // 3. Handle new tracks
    for (size_t d = 0; d < detections.size(); ++d) {
        if (!det_matched[d] && detections[d].score >= new_track_thresh) {
             Track t = detections[d];
             t.track_id = next_id++;
             t.last_seen = std::chrono::system_clock::now();
             tracked_stracks.push_back(t);
             if (debug_frame < 50) {
                 std::cerr << "[TRACKER] NEW track=" << t.track_id 
                           << " score=" << t.score << " bbox=[" 
                           << t.bbox[0] << "," << t.bbox[1] << "," 
                           << t.bbox[2] << "," << t.bbox[3] << "]" << std::endl;
             }
        } else if (!det_matched[d]) {
             if (debug_frame < 50) {
                 std::cerr << "[TRACKER] LOW_CONF det_score=" << detections[d].score 
                           << " (need >= " << new_track_thresh << ")" << std::endl;
             }
        }
    }
    
    // 4. Handle lost tracks
    // Remove tracks that haven't been updated for too long
    auto it = std::remove_if(tracked_stracks.begin(), tracked_stracks.end(), 
        [this](const Track& t){ return t.time_since_update > this->max_time_lost; });
    tracked_stracks.erase(it, tracked_stracks.end());
    
    // Only return tracks that were matched/updated in THIS frame.
    // Returning stale tracks (time_since_update > 0) causes wrong crops:
    // their bbox is from a previous frame, applied to current frame data.
    std::vector<Track> active;
    active.reserve(tracked_stracks.size());
    for (const auto& tr : tracked_stracks) {
        if (tr.time_since_update == 0) active.push_back(tr);
    }
    return active;
}
