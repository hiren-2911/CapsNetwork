import matplotlib.pyplot as plt

def extract():
    file=open("/home/ml/Hiren/Code/CycleCNN/trial_Test1_2022-09-28_20_7/train_log_Test1_2022-09-28_20_7.txt",'r')
    perc=[]
    mse=[]
    for line in file:
        #print(line)
        perc.append(float(line.split(',')[2].split(':')[1]))
        mse.append(float(line.split(',')[5].split(':')[1]))

    print(sum(perc)/len(perc))
    print(sum(mse)/len(mse))

def extract_loss():
    file=open("/home/ml/Hiren/Code/CycleCNN/trial_Test1_2022-10-12_0_16/train_log_Test1_2022-10-12_0_16_train_loss.txt",'r')
    loss=[]
    for idx,line in enumerate(file):
        if(idx==0):
            continue
        loss.append(float(line.split(':')[1]))

    plt.plot(loss[1:])

    #plt.plot(perc[2:])
    #plt.plot(mse[1:])
    plt.show()
    #print(sum(perc)/len(perc))
    #print(sum(mse)/len(mse))




if __name__=="__main__":
    extract_loss()
