#include <single_joint_episode_generator/LadderMode.h>

LadderMode::LadderMode(float amplitude, float start_time) {
    _amplitude = amplitude;
    _start_time = start_time;
}

float LadderMode::getAmplitude() {
    return _amplitude;
}

void LadderMode::setAmplitude(float amplitude) {
    _amplitude = amplitude;
}

int LadderMode::getStartTime() {
    return _start_time;
}

void LadderMode::setStartTime(int start_time) {
    _start_time = start_time;
}

float LadderMode::sample(int t) {
    if (t < _start_time) {
        return 0.0;
    } else {
        return _amplitude;
    }
}
