#  json2md.py  

Convert benchmark json output to Markdown

## Usage:  

Assuming you are in the *build* folder

```
./benchmarks/bvlc_reference_caffenet --benchmark_format=json | tail -n +2 > out.json
./json2md.py -i out.json -o out.md
```

## Example output:

