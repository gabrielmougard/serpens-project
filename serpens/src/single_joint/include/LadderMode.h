class LadderMode {

    private:
        float _amplitude;
        int _start_time;

    public:
        LadderMode(float amplitude, float start_time);
        ~LadderMode();
        void setAmplitude(float amplitude);
        float getAmplitude();
        void setStartTime(int start_time);
        int getStartTime();
        float sample();
}