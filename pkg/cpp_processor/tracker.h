#ifndef TRACKER_H
#define TRACKER_H

#include <vector>
#include <map>
#include <deque>
#include <string>
#include <chrono>

// Track struct to hold state for a single object
struct Track {
    int track_id = -1;
    int class_id = -1;
    float score = 0.0f;
    std::vector<float> features; // Re-ID features (if used)
    
    // Kalman Filter State (Simplified: [x, y, w, h, vx, vy, vw, vh])
    // Or just [x, y, w, h] for very basic overlap matching
    float bbox[4] = {0.0f, 0.0f, 0.0f, 0.0f}; // xywh
    
    int time_since_update = 0;
    std::chrono::system_clock::time_point last_seen;
};

class BoTSORT {
public:
    BoTSORT(float track_high_thresh = 0.6, float track_low_thresh = 0.1, 
            float new_track_thresh = 0.7, int track_buffer = 30, float match_thresh = 0.8);
    
    // Update tracker with new detections
    // input_dets: [class_id, confidence, x, y, w, h]
    std::vector<Track> update(const std::vector<std::vector<float>>& input_dets);

private:
    int frame_id;
    int max_time_lost;
    float track_high_thresh;
    float track_low_thresh;
    float new_track_thresh;
    float match_thresh;
    
    int next_id;
    
    std::vector<Track> tracked_stracks;
    std::vector<Track> lost_stracks;
    std::vector<Track> removed_stracks;

    // Helper: Calculate IoU between two boxes
    float iou(const float* box1, const float* box2);
};

#endif // TRACKER_H
