#!/usr/bin/python
#
# json2md.py
#
# Converts google-benchmark output from json to Markdown

import sys, os, getopt, json

def main(argv):
     layer_names = ['**conv1**',
                    '**relu1**',
                    '**pool1**',
                    '**lrn1**',
                    '**conv2**',
                    '**relu2**',
                    '**pool2**',
                    '**lrn2**',
                    '**conv3**',
                    '**relu3**',
                    '**conv4**',
                    '**relu4**',
                    '**conv5**',
                    '**relu5**',
                    '**pool5**',
                    '**fc6**',
                    '**relu6**',
                    '**drop6**',
                    '**fc7**',
                    '**relu7**',
                    '**drop7**',
                    '**fc8**',
                    '**prob**']

     # Parse command line arguments
     script_name = argv[0]
     in_file_path = ''
     out_file_path = ''
     try:
         opts, args = getopt.getopt(argv[1:],"hi:o:",["ifile=","ofile="])
     except getopt.GetoptError:
         print script_name, '-i <in_file_path> -o <out_file_path>'
         sys.exit(2)

     if len(opts) < 2:
         print script_name, '-i <in_file_path> -o <out_file_path>'
         sys.exit()

     for opt, arg in opts:
         if opt == '-h':
             print script_name, '-i <in_file_path> -o <out_file_path>'
             sys.exit()
         elif opt in ("-i", "--ifile"):
             if not os.path.isfile(arg):
                 print 'Input file not found or not a regular file:', arg
                 sys.exit(2)
             in_file_path = arg
         elif opt in ("-o", "--ofile"):
             out_dir = os.path.dirname(arg)
             if out_dir != '' and not os.path.exists(out_dir):
                 print 'Output dir not found:', os.path.dirname(arg)
                 sys.exit(2)
             out_file_path = arg
         else:
             print 'Unknown option', opt
             sys.exit(2)
  
     print 'Converting', in_file_path,'to', out_file_path
 
     # Parse json input 
     with open(in_file_path) as in_file:    
         data = json.load(in_file)

     # Open md file output
     out_file = open(out_file_path, 'w')

     # Separate caffe benchmarks from tiny-dnn
     caffe_benchmarks = []
     tiny_dnn_benchmarks = []
     for benchmark in data['benchmarks']:
         if 'CaffeLayerTest' in benchmark['name']:
             caffe_benchmarks.append(benchmark)
         elif 'TinyDNNLayerTest' in benchmark['name']:
             tiny_dnn_benchmarks.append(benchmark)
     
     # Validate number of Caffe and tiny-dnn benchmarks matches
     if len(caffe_benchmarks) != len(tiny_dnn_benchmarks):
         print 'Error: number of Caffe and tiny-dnn benchmarks must match' 
         print 'Caffe =', len(caffe_benchmarks), 'tiny-dnn =', len(tiny_dnn_benchmarks)
         sys.exit(2)

     # Write header
     c = data['context']
     out_file.write('### ' + caffe_benchmarks[0]['name'].split('/')[0] + ':\n-\n')
     out_file.write('Date: **' + c['date'] + '**  \n')
     out_file.write('Threads: ' + "{0:.4f}".format(c['num_cpus']) + ' @ ' + "{0:.4f}".format(c['mhz_per_cpu']) + ' Mhz  \n')
     out_file.write('Build: ' + c['library_build_type'] + '  \n\n')

     # Write benchmarks into a markdown table
     out_file.write('| Layer | Caffe CPU | tiny-dnn CPU | Caffe time | tiny-dnn time |\n')
     out_file.write(':---:| ---:| ---:| ---:| ---:\n')
     for c, t in zip(caffe_benchmarks, tiny_dnn_benchmarks):
         caffe_layer_idx = int(c['name'].split('/')[-1])
         tiny_dnn_layer_idx = int(t['name'].split('/')[-1])

         if caffe_layer_idx != tiny_dnn_layer_idx:
             print 'Error: layer index of Caffe and tiny-dnn must match' 
             print 'Caffe =', caffe_layer_idx, 'tiny-dnn =', tiny_dnn_layer_idx 
             sys.exit(2)
         out_file.write(layer_names[caffe_layer_idx-1] + ' | ' + "{0:.4f}".format(c['cpu_time'] / 1000000.0) + ' ms | ' + "{0:.4f}".format(t['cpu_time'] / 1000000.0) + ' ms | ' + "{0:.4f}".format(c['real_time'] / 1000000.0) + ' ms | ' + "{0:.4f}".format(t['real_time'] / 1000000.0) + ' ms\n')

if __name__ == '__main__':
     main(sys.argv)
