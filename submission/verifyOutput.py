import os, random, subprocess

# random.seed(2)
fname_ls = ["outputData.txt"]
algo_dict = {"outputData.txt": ['epsilon-greedy-t1', 'ucb-t1', 'kl-ucb-t1', 'thompson-sampling-t1', 'ucb-t2', 'alg-t3', 'alg-t4']}
number_lines = 9150
for fname in fname_ls:
    errorFlag = False

    #Check if file exists
    print('\n-------- verifying', fname, 'data ---------------')
    try:
        f = open(fname, "r")
        line_ls = [line.strip() for line in f.readlines()]
        len_line_ls = len(line_ls)

        # Check for number of lines
        if not (len_line_ls == number_lines or len_line_ls == number_lines + 1):
            print("\n", "*" * 10, "Mistake:number of lines in the output data file should be", number_lines, "but have ", len_line_ls, "*" * 10, "\n")
            errorFlag = True

        lists = []
        set_main = set()
        for i in range(9):
            lists.append([])

        for line in line_ls:
            if ", " in line:
                line = line.replace("\n", "").split(", ")
            elif "," in line:
                line = line.replace("\n", "").split(",")
            if not len(line) == 9:
                print("\n", "*" * 10, "Mistake: Wrong line printed", line, "*" * 10, "\n")
                continue
            lists[0].append(line[0])  #instance
            lists[1].append(line[1])  #algo
            lists[2].append(int(line[2]))  #randomSeed
            lists[3].append(float(line[3]))  #epsilon
            lists[4].append(float(line[4]))  #scale
            lists[5].append(float(line[5]))  #threshold
            lists[6].append(int(line[6]))  # horizon
            lists[7].append(float(line[7]))  # REG
            lists[8].append(int(line[8]))  # HIGHS

            set_main.add(line[0] + "--" + line[1] + "--" + line[2] + "--" + line[3] + "--" + line[4] + "--" + line[5] + "--" + line[6])

        if not len(set_main) == number_lines:
            print("\n", "*" * 10, "Mistake: You didn't print all the combinations. Need ", number_lines, "but printed ", len(set_main), "*" * 10, "\n")
            errorFlag = True

        #try to reproduce random 10 data points
        for i in range(10):
            idx = random.randint(0, len_line_ls)
            line_str = line_ls[idx]
            if ", " in line_str:
                line = line_str.replace("\n", "").split(", ")
            elif "," in line_str:
                line = line_str.replace("\n", "").split(",")
            orig_REG = line[-2].strip()
            orig_HIGHS = line[-1].strip()

            cmd = "python", "bandit.py", "--instance", line[0].strip(), "--algorithm", line[1].strip(), "--randomSeed", line[2].strip(), "--epsilon", line[3].strip(), "--scale", line[4].strip(), "--threshold", line[5].strip(), "--horizon", line[6].strip()
            print("running", ' '.join(cmd))
            reproduced_str = subprocess.check_output(cmd, universal_newlines=True)
            reproduced = reproduced_str.replace("\n", "").split(",")
            rep_REG = reproduced[-2].strip()
            rep_HIGH = reproduced[-1].strip()

            if not rep_REG == orig_REG:
                print("\n", "*" * 10, "Mistake: Unable to reproduce result for REG ", line_str, " orignal=" + orig_REG + " reproduced=" + rep_REG, "\t", "*" * 10, "\n")
                errorFlag = True
            if not orig_HIGHS == rep_HIGH:
                print("\n", "*" * 10, "Mistake: Unable to reproduce result for HIGHS ", line_str, " orignal=" + orig_HIGHS + " reproduced=" + rep_HIGH, "\t", "*" * 10, "\n")
                errorFlag = True
        f.close()
    except Exception as e:
        print("*" * 10, "Mistake:There is no file named", fname, "*" * 10)
        print(e)
        errorFlag = True

if errorFlag:
    print("\n", "*" * 10, "Some issue with your submission data", "*" * 10, "\n")
else:
    print("Everything is Okay")
