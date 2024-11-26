

#coding:utf-8

import numpy as np
import csv
import time
global label_list  #label_list为全局变量

#定义kdd99数据预处理函数
def preHandel_data():
    source_file='KDDTest-21'
    handled_file='KDDTest-21_processed.csv'
    data_file=open(handled_file,'w',newline='')     #python3.x中添加newline=''这一参数使写入的文件没有多余的空行
    with open(source_file,'r') as data_source:
        csv_reader=csv.reader(data_source)
        csv_writer=csv.writer(data_file)
        count=0   #记录数据的行数，初始化为0
        for row in csv_reader:
            temp_line=np.array(row)   #将每行数据存入temp_line数组里
            temp_line[1]=handleProtocol(row)   #将源文件行中3种协议类型转换成数字标识
            temp_line[2]=handleService(row)    #将源文件行中70种网络服务类型转换成数字标识
            temp_line[3]=handleFlag(row)       #将源文件行中11种网络连接状态转换成数字标识
            temp_line[41]=handleLabel(row)   #将源文件行中23种攻击类型转换成数字标识
            csv_writer.writerow(temp_line)
            count+=1
            #输出每行数据中所修改后的状态
            print(count,'status:',temp_line[1],temp_line[2],temp_line[3],temp_line[41])
        data_file.close()



#Converts corresponding non-numeric types to numeric identifiers, i.e. symbolic data to numeric data
def find_index(x,y):
    return [i for i in range(len(y)) if y[i]==x]

#Defines a function that converts the three protocol types in a source file line into a numeric identifier
def handleProtocol(input):
    protocol_list=['tcp','udp','icmp']
    if input[1] in protocol_list:
        return find_index(input[1],protocol_list)[0]

#Defines a function that converts 70 network service types in a source file line to a numeric identifier
def handleService(input):
   service_list=['aol','auth','bgp','courier','csnet_ns','ctf','daytime','discard','domain','domain_u',
                 'echo','eco_i','ecr_i','efs','exec','finger','ftp','ftp_data','gopher','harvest','hostnames',
                 'http','http_2784','http_443','http_8001','imap4','IRC','iso_tsap','klogin','kshell','ldap',
                 'link','login','mtp','name','netbios_dgm','netbios_ns','netbios_ssn','netstat','nnsp','nntp',
                 'ntp_u','other','pm_dump','pop_2','pop_3','printer','private','red_i','remote_job','rje','shell',
                 'smtp','sql_net','ssh','sunrpc','supdup','systat','telnet','tftp_u','tim_i','time','urh_i','urp_i',
                 'uucp','uucp_path','vmnet','whois','X11','Z39_50']
   if input[2] in service_list:
       return find_index(input[2],service_list)[0]

#Defines a function to convert the 11 network connection states in the source file rows into numeric identifiers
def handleFlag(input):
    flag_list=['OTH','REJ','RSTO','RSTOS0','RSTR','S0','S1','S2','S3','SF','SH']
    if input[3] in flag_list:
        return find_index(input[3],flag_list)[0]

#Define a function that converts the attack types in the source file lines to numeric identifiers (22 attack types occur in the training set, while the remaining 17 occur only in the test set)
def handleLabel(input):
    #label_list=['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
    # 'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
    # 'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
    # 'spy.', 'rootkit.']
    global label_list
    if input[41] in label_list:
        return find_index(input[41],label_list)[0]
    else:
        label_list.append(input[41])
        return find_index(input[41],label_list)[0]


if __name__=='__main__':
    start_time=time.perf_counter()
    global label_list   #声明一个全局变量的列表并初始化为空
    label_list=[]
    preHandel_data()
    end_time=time.perf_counter()
    print("Running time:",(end_time-start_time))  #输出程序运行时间

