#!/bin/bash
set -e
OUTPUT="/Volumes/Extreme SSD/stems/separated_stems_nocorrection"
DONE=0
FAIL=0
TOTAL=300

echo "[1/300] Abhogi_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_1.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[2/300] Abhogi_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[3/300] Abhogi_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_3.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[4/300] Abhogi_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[5/300] Abhogi_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[6/300] Abhogi_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[7/300] Abhogi_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[8/300] Abhogi_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[9/300] Abhogi_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Abhogi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[10/300] Abhogi_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Abhogi/Abhogi_10.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Abhogi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[11/300] Ahir Bhairav_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_1.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[12/300] Ahir Bhairav_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_2.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[13/300] Ahir Bhairav_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[14/300] Ahir Bhairav_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[15/300] Ahir Bhairav_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[16/300] Ahir Bhairav_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[17/300] Ahir Bhairav_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_7.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[18/300] Ahir Bhairav_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[19/300] Ahir Bhairav_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[20/300] Ahir Bhairav_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Ahir Bhairav/Ahir Bhairav_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Ahir Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[21/300] Alhaiya Bilaval_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[22/300] Alhaiya Bilaval_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[23/300] Alhaiya Bilaval_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_3.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[24/300] Alhaiya Bilaval_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[25/300] Alhaiya Bilaval_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[26/300] Alhaiya Bilaval_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[27/300] Alhaiya Bilaval_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[28/300] Alhaiya Bilaval_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[29/300] Alhaiya Bilaval_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_9.mp3" \
  --output "$OUTPUT" \
  --tonic "G" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[30/300] Alhaiya Bilaval_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Alhaiya Bilawal/Alhaiya Bilaval_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Alhaiya Bilawal" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[31/300] Bageshri_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[32/300] Bageshri_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[33/300] Bageshri_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[34/300] Bageshri_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[35/300] Bageshri_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[36/300] Bageshri_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[37/300] Bageshri_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[38/300] Bageshri_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[39/300] Bageshri_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Bageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[40/300] Bageshri_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bageshri/Bageshri_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[41/300] Bairagi_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[42/300] Bairagi_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_2.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bairagi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[43/300] Bairagi_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[44/300] Bairagi_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_4.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[45/300] Bairagi_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[46/300] Bairagi_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[47/300] Bairagi_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[48/300] Bairagi_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[49/300] Bairagi_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[50/300] Bairagi_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bairagi/Bairagi_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bairagi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[51/300] Basant_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_1.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[52/300] Basant_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_2.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Basant" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[53/300] Basant_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[54/300] Basant_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[55/300] Basant_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Basant" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[56/300] Basant_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Basant" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[57/300] Basant_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_7.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Basant" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[58/300] Basant_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[59/300] Basant_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[60/300] Basant_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Basant/Basant_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Basant" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[61/300] Bhairav_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_1.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Bhairav" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[62/300] Bhairav_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[63/300] Bhairav_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[64/300] Bhairav_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[65/300] Bhairav_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[66/300] Bhairav_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[67/300] Bhairav_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[68/300] Bhairav_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_8.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[69/300] Bhairav_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_9.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[70/300] Bhairav_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhairav/Bhairav_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhairav" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[71/300] Bhupali_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[72/300] Bhupali_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[73/300] Bhupali_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[74/300] Bhupali_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[75/300] Bhupali_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[76/300] Bhupali_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[77/300] Bhupali_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[78/300] Bhupali_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[79/300] Bhupali_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_9.mp3" \
  --output "$OUTPUT" \
  --tonic "G" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[80/300] Bhupali_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bhupali/Bhupali_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bhupali" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[81/300] Bihag_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_1.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[82/300] Bihag_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_2.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[83/300] Bihag_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[84/300] Bihag_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[85/300] Bihag_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[86/300] Bihag_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[87/300] Bihag_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[88/300] Bihag_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[89/300] Bihag_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[90/300] Bihag_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bihag/Bihag_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[91/300] Bilaskhani Todi_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_1.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[92/300] Bilaskhani Todi_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[93/300] Bilaskhani Todi_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[94/300] Bilaskhani Todi_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[95/300] Bilaskhani Todi_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[96/300] Bilaskhani Todi_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[97/300] Bilaskhani Todi_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_7.mp3" \
  --output "$OUTPUT" \
  --tonic "G#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[98/300] Bilaskhani Todi_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[99/300] Bilaskhani Todi_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[100/300] Bilaskhani Todi_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Bilaskhani Todi/Bilaskhani Todi_10.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Bilaskhani Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[101/300] Darbari Kanada_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_1.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[102/300] Darbari Kanada_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[103/300] Darbari Kanada_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[104/300] Darbari Kanada_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[105/300] Darbari Kanada_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[106/300] Darbari Kanada_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[107/300] Darbari Kanada_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[108/300] Darbari Kanada_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[109/300] Darbari Kanada_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[110/300] Darbari Kanada_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Darbari Kanada/Darbari Kanada_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Darbari Kanada" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[111/300] Desh_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_1.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[112/300] Desh_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[113/300] Desh_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[114/300] Desh_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[115/300] Desh_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Desh" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[116/300] Desh_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[117/300] Desh_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[118/300] Desh_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[119/300] Desh_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[120/300] Desh_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Desh/Desh_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Desh" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[121/300] Gaud Malhar_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_1.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[122/300] Gaud Malhar_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_2.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[123/300] Gaud Malhar_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[124/300] Gaud Malhar_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[125/300] Gaud Malhar_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[126/300] Gaud Malhar_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[127/300] Gaud Malhar_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_7.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[128/300] Gaud Malhar_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[129/300] Gaud Malhar_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[130/300] Gaud Malhar_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Gaud Malhar/Gaud Malhar_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Gaud Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[131/300] Hansadhwani_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[132/300] Hansadhwani_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_2.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[133/300] Hansadhwani_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[134/300] Hansadhwani_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[135/300] Hansadhwani_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[136/300] Hansadhwani_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[137/300] Hansadhwani_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[138/300] Hansadhwani_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[139/300] Hansadhwani_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[140/300] Hansadhwani_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Hansadhwani/Hansadhwani_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Hansadhwani" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[141/300] Jog_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_1.mp3" \
  --output "$OUTPUT" \
  --tonic "" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[142/300] Jog_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[143/300] Jog_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[144/300] Jog_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[145/300] Jog_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[146/300] Jog_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[147/300] Jog_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[148/300] Jog_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[149/300] Jog_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Jog" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[150/300] Jog_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Jog/Jog_10.mp3" \
  --output "$OUTPUT" \
  --tonic "G#" --raga "Jog" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[151/300] Kedar_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[152/300] Kedar_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_2.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[153/300] Kedar_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[154/300] Kedar_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[155/300] Kedar_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[156/300] Kedar_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[157/300] Kedar_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[158/300] Kedar_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[159/300] Kedar_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[160/300] Kedar_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Kedar/Kedar_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Kedar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[161/300] Khamaj_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_1.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[162/300] Khamaj_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[163/300] Khamaj_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[164/300] Khamaj_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[165/300] Khamaj_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[166/300] Khamaj_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[167/300] Khamaj_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_7.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[168/300] Khamaj_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[169/300] Khamaj_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[170/300] Khamaj_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Khamaj/Khamaj_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Khamaj" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[171/300] Lalit_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[172/300] Lalit_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_2.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Lalit" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[173/300] Lalit_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Lalit" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[174/300] Lalit_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_4.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[175/300] Lalit_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_5.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[176/300] Lalit_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[177/300] Lalit_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[178/300] Lalit_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[179/300] Lalit_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[180/300] Lalit_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Lalit/Lalit_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Lalit" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[181/300] Madhukauns_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_1.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[182/300] Madhukauns_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[183/300] Madhukauns_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[184/300] Madhukauns_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[185/300] Madhukauns_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[186/300] Madhukauns_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[187/300] Madhukauns_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_7.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[188/300] Madhukauns_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_8.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[189/300] Madhukauns_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[190/300] Madhukauns_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhukauns/Madhukauns_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Madhukauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[191/300] Madhuvanti_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[192/300] Madhuvanti_11..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_11.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[193/300] Madhuvanti_12..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_12.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[194/300] Madhuvanti_13..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_13.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[195/300] Madhuvanti_14..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_14.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[196/300] Madhuvanti_15..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_15.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[197/300] Madhuvanti_16..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_16.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[198/300] Madhuvanti_17..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_17.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[199/300] Madhuvanti_18..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_18.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[200/300] Madhuvanti_19..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Madhuvanti/Madhuvanti_19.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Madhuvanti" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[201/300] Malkauns_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_1.mp3" \
  --output "$OUTPUT" \
  --tonic "G" --raga "Malkauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[202/300] Malkauns_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[203/300] Malkauns_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[204/300] Malkauns_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[205/300] Malkauns_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[206/300] Malkauns_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[207/300] Malkauns_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[208/300] Malkauns_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_8.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Malkauns" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[209/300] Malkauns_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[210/300] Malkauns_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Malkauns/Malkauns_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Malkauns" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[211/300] Maru Bihag_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_1.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[212/300] Maru Bihag_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[213/300] Maru Bihag_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[214/300] Maru Bihag_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[215/300] Maru Bihag_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[216/300] Maru Bihag_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[217/300] Maru Bihag_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[218/300] Maru Bihag_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_8.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[219/300] Maru Bihag_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[220/300] Maru Bihag_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Maru Bihag/Maru Bihag_10.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Maru Bihag" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[221/300] Marva_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[222/300] Marva_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[223/300] Marva_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[224/300] Marva_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[225/300] Marva_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[226/300] Marva_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[227/300] Marva_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[228/300] Marva_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[229/300] Marva_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Marwa" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[230/300] Marva_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Marwa/Marva_10.mp3" \
  --output "$OUTPUT" \
  --tonic "G#" --raga "Marwa" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[231/300] Miyan ki Malhar_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_1.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[232/300] Miyan ki Malhar_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_2.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[233/300] Miyan ki Malhar_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_3.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[234/300] Miyan ki Malhar_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_4.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[235/300] Miyan ki Malhar_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[236/300] Miyan ki Malhar_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[237/300] Miyan ki Malhar_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[238/300] Miyan ki Malhar_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[239/300] Miyan ki Malhar_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[240/300] Miyan ki Malhar_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Miyan ki Malhar/Miyan ki Malhar_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Miyan ki Malhar" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[241/300] Puriya Dhanashri_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[242/300] Puriya Dhanashri_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[243/300] Puriya Dhanashri_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_3.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[244/300] Puriya Dhanashri_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_4.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[245/300] Puriya Dhanashri_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[246/300] Puriya Dhanashri_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[247/300] Puriya Dhanashri_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[248/300] Puriya Dhanashri_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_8.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[249/300] Puriya Dhanashri_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_9.mp3" \
  --output "$OUTPUT" \
  --tonic "G#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[250/300] Puriya Dhanashri_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Puriya Dhanashree/Puriya Dhanashri_10.mp3" \
  --output "$OUTPUT" \
  --tonic "G#" --raga "Puriya Dhanashree" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[251/300] Rageshri_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_1.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[252/300] Rageshri_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[253/300] Rageshri_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_3.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[254/300] Rageshri_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Rageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[255/300] Rageshri_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[256/300] Rageshri_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[257/300] Rageshri_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_7.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[258/300] Rageshri_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_8.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[259/300] Rageshri_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_9.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[260/300] Rageshri_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Rageshri/Rageshri_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Rageshri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[261/300] Shri_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[262/300] Shri_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_2.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[263/300] Shri_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[264/300] Shri_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[265/300] Shri_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_5.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[266/300] Shri_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_6.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[267/300] Shri_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[268/300] Shri_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[269/300] Shri_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_9.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[270/300] Shri_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shri/Shri_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shri" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[271/300] Shuddh Sarang_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[272/300] Shuddh Sarang_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_2.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[273/300] Shuddh Sarang_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[274/300] Shuddh Sarang_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_4.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[275/300] Shuddh Sarang_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_5.mp3" \
  --output "$OUTPUT" \
  --tonic "E" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[276/300] Shuddh Sarang_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_6.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[277/300] Shuddh Sarang_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_7.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[278/300] Shuddh Sarang_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[279/300] Shuddh Sarang_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_9.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[280/300] Shuddh Sarang_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Shuddh Sarang/Shuddh Sarang_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Shuddh Sarang" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[281/300] Todi_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[282/300] Todi_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_2.mp3" \
  --output "$OUTPUT" \
  --tonic "F" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[283/300] Todi_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_3.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[284/300] Todi_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[285/300] Todi_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_5.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[286/300] Todi_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_6.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[287/300] Todi_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_7.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Todi" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[288/300] Todi_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_8.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[289/300] Todi_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_9.mp3" \
  --output "$OUTPUT" \
  --tonic "F#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[290/300] Todi_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Todi/Todi_10.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Todi" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[291/300] Yaman Kalyan_1..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_1.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[292/300] Yaman Kalyan_2..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_2.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[293/300] Yaman Kalyan_3..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_3.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[294/300] Yaman Kalyan_4..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_4.mp3" \
  --output "$OUTPUT" \
  --tonic "D" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[295/300] Yaman Kalyan_5..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_5.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[296/300] Yaman Kalyan_6..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_6.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[297/300] Yaman Kalyan_7..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_7.mp3" \
  --output "$OUTPUT" \
  --tonic "A#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[298/300] Yaman Kalyan_8..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_8.mp3" \
  --output "$OUTPUT" \
  --tonic "D#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[299/300] Yaman Kalyan_9..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_9.mp3" \
  --output "$OUTPUT" \
  --tonic "A" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender female \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "[300/300] Yaman Kalyan_10..."
python driver.py analyze \
  --audio "/Volumes/Extreme SSD/RagaDataset/compmusic/Yaman Kalyan/Yaman Kalyan_10.mp3" \
  --output "$OUTPUT" \
  --tonic "C#" --raga "Yaman Kalyan" \
  --source-type vocal --vocalist-gender male \
  --skip-raga-correction > /dev/null 2>&1 && DONE=$((DONE+1)) || FAIL=$((FAIL+1))

echo "Batch complete: $DONE succeeded, $FAIL failed out of $TOTAL"
