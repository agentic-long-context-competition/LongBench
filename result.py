import os, json

files = os.listdir('results')
output = ["Model\tOverall\tEasy\tHard\tShort\tMedium\tLong\tErrors"]
compensated = False

for file in files:
    filename = os.path.join('results', file)
    try:
        pred_data = json.load(open(filename, encoding='utf-8'))
    except Exception as e:
        pred_data = [json.loads(line) for line in open(filename, encoding='utf-8')]
    easy, hard, short, medium, long = 0, 0, 0, 0, 0
    easy_acc, hard_acc, short_acc, medium_acc, long_acc = 0, 0, 0, 0, 0
    error_count = 0  # Initialize error count
    
    for pred in pred_data:
        if 'error' in pred:
            error_count += 1  # Count errors
            continue

        acc = int(pred['judge'])
        if compensated and pred["pred"] == None:
            acc = 0.25
        if pred["difficulty"] == "easy":
            easy += 1
            easy_acc += acc
        else:
            hard += 1
            hard_acc += acc

        if pred['length'] == "short":
            short += 1
            short_acc += acc
        elif pred['length'] == "medium":
            medium += 1
            medium_acc += acc
        else:
            long += 1
            long_acc += acc

    name = '.'.join(file.split('.')[:-1])
    num_successful_queries = easy + hard
    total_queries = num_successful_queries + error_count
    output.append(name+'\t'+
                  str(round(100*(easy_acc+hard_acc)/num_successful_queries, 1) if num_successful_queries != 0 else 'nan')+'\t'+
                  str(round(100*easy_acc/easy, 1) if easy != 0 else 'nan')+'\t'+
                  str(round(100*hard_acc/hard, 1) if hard != 0 else 'nan')+'\t'+
                  str(round(100*short_acc/short, 1) if short != 0 else 'nan')+'\t'+
                  str(round(100*medium_acc/medium, 1) if medium != 0 else 'nan')+'\t'+
                  str(round(100*long_acc/long, 1) if long != 0 else 'nan')+'\t'+
                  str(error_count))

open('result.txt', 'w', encoding='utf-8').write('\n'.join(output))
