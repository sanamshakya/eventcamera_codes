# Capture RAW stream along with event stream :
# with events csv capture for raw capture pipeline along with raw input capture
TS=$(date +%Y%m%d_%H%M%S)

timeout 30s ./sample_camera --rig rig_raw.json --enable-events=1 --event-display=1 \
    --event-input=raw --event-deterministic=0 --event-max=20000000 \
    --event-sensor=custom_sensor_raw --event-clahe=1 --event-height=720 \
    --event-width=1280 --event-csv=clahe_raw_custom_sensor \
    --dump-input=clahe_raw_input_${TS} --dump-input-streams=raw \
    --dump-input-every=1 --dump-input-max=100