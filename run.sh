#!/bin/bash 
echo "runtime | driver"> comparison_output.txt
for i in {1..5}
do
./runtime >> comparison_output.txt && ./fixed_low_level_3mm.exe >> comparison_output.txt
echo " " >> comparison_output.txt
done

echo "driver | runtime " >> comparison_output.txt
for i in {1..5}
do
./fixed_low_level_3mm.exe >> comparison_output.txt && ./runtime >> comparison_output.txt
echo " " >> comparison_output.txt
done
cat comparison_output.txt