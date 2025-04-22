#!/bin/bash

# Set PCM controls for capture
sudo amixer -c 0 cset numid=31 'Analogue 1'  # 'PCM 01' - Set to 'Analogue 1'
sudo amixer -c 0 cset numid=32 'Analogue 1'  # 'PCM 02' - Set to 'Analogue 1'
sudo amixer -c 0 cset numid=33 'Off'         # 'PCM 03' - Disabled
sudo amixer -c 0 cset numid=34 'Off'         # 'PCM 04' - Disabled

# Set DSP Input controls (Unused, set to Off)
sudo amixer -c 0 cset numid=29 'Off'         # 'DSP Input 1'
sudo amixer -c 0 cset numid=30 'Off'         # 'DSP Input 2'

# Configure Line In 1 as main input for mono setup
sudo amixer -c 0 cset numid=8 'Off'          # 'Line In 1 Air' - Keep 'Off'
sudo amixer -c 0 cset numid=14 off           # 'Line In 1 Autogain' - Disabled
sudo amixer -c 0 cset numid=6 'Line'         # 'Line In 1 Level' - Set level to 'Line'
sudo amixer -c 0 cset numid=21 on           # 'Line In 1 Safe' - Enabled to avoid clipping / noise impact ?

# Disable Line In 2 to minimize interference (if not used)
sudo amixer -c 0 cset numid=9 'Off'          # 'Line In 2 Air'
sudo amixer -c 0 cset numid=17 off           # 'Line In 2 Autogain' - Disabled
sudo amixer -c 0 cset numid=16 0             # 'Line In 2 Gain' - Set gain to 0 (mute)
sudo amixer -c 0 cset numid=7 'Line'         # 'Line In 2 Level' - Set to 'Line'
sudo amixer -c 0 cset numid=22 off           # 'Line In 2 Safe' - Disabled

# Set Line In 1-2 controls
sudo amixer -c 0 cset numid=12 off           # 'Line In 1-2 Link' - No need to link for mono
sudo amixer -c 0 cset numid=10 on            # 'Line In 1-2 Phantom Power' - Enabled for condenser mics

# Set Analogue Outputs to use the same mix for both channels (Mono setup)
sudo amixer -c 0 cset numid=23 'Mix A'       # 'Analogue Output 01' - Set to 'Mix A'
sudo amixer -c 0 cset numid=24 'Mix A'       # 'Analogue Output 02' - Same mix as Output 01

# Set Direct Monitor to off to prevent feedback
sudo amixer -c 0 cset numid=53 'Off'         # 'Direct Monitor'

# Set Input Select to Input 1
sudo amixer -c 0 cset numid=11 'Input 1'     # 'Input Select'

# Optimize Monitor Mix settings for mono output
sudo amixer -c 0 cset numid=54 153           # 'Monitor 1 Mix A Input 01' - Set to 153 (around -3.50 dB)
sudo amixer -c 0 cset numid=55 153           # 'Monitor 1 Mix A Input 02' - Set to 153 for balanced output
sudo amixer -c 0 cset numid=56 0             # 'Monitor 1 Mix A Input 03' - Mute unused channels
sudo amixer -c 0 cset numid=57 0             # 'Monitor 1 Mix A Input 04'

# Set Sync Status to Locked
sudo amixer -c 0 cset numid=52 'Locked'      # 'Sync Status'

echo "Mono optimization applied. Only using primary input and balanced outputs."
