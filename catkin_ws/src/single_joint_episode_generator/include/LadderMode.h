class LadderMode {

    private:
        float _amplitude;
        float _start_time;

    public:
        LadderMode(float amplitude, float start_time);
        ~LadderMode();
        float sample();
}