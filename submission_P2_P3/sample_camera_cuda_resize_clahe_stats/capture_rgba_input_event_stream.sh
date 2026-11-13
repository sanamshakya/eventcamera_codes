# Capture RGBA stream along with event stream :
#with events csv capture logging for rgba clahe along with rgba input capture
TS=$(date +%Y%m%d_%H%M%S)

timeout 30s ./sample_camera --rig rig.json --enable-events=1 --event-display=1 \
    --event-input=rgba --event-deterministic=0 --event-max=20000000 \
    --event-sensor=custom_sensor_rrba --event-clahe=1 --event-height=720 \
    --event-width=1280 --event-csv=clahe_rgba_custom_sensor \
    --dump-input=clahe_rgba_input_${TS} --dump-input-streams=rgba \
    --dump-input-every=1 --dump-input-max=100