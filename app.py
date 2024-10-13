import streamlit as st 
from transformers import AutoTokenizer,TFBertModel
import pandas as pd
import numpy as np
import PyPDF2
import re
import pdfplumber
from textrank4zh import TextRank4Sentence

def dg_get(fm):
    ct=str()#论文全部内容
    list1_dg=[]#大纲条数
    with open(fm, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for  page in reader.pages:
            text=page.extract_text()
            ct=ct+'，'+text
            lines=text.split('\n')
            for line in lines:
                line1=line.replace( '', '')
                #对每一个
                if len(line1)<=20 and len(line1)>3:
                    if line1[:2]=='一、' or line1[:2]=='二、' or line1[:2]=='三、' or line1[:2]=='四、' or line1[:2]=='五、'  or line1[:2]=='六、'  or line1[:2]=='七、' or line1[:2]=='八、' or line1[:2]=='九、' or line1[:2]=='十、':
                         list1_dg.append(line1)  
                    elif line1[:2]=='一 ' or line1[:2]=='二 ' or line1[:2]=='三 ' or line1[:2]=='四 ' or line1[:2]=='五 '  or line1[:2]=='六 '  or line1[:2]=='七 ' or line1[:2]=='八 ' or line1[:2]=='九 ' or line1[:2]=='十 ':
                              list1_dg.append(line1)  
                    elif  line1[:2]=='（一' or line1[:2]=='（二' or line1[:2]=='（三' or line1[:2]=='（四' or line1[:2]=='（五'  or line1[:2]=='（六'  or line1[:2]=='（七' or line1[:2]=='（八' or line1[:2]=='（九' or line1[:2]=='（十'  :
                          list1_dg.append(line1)
                    elif   line1[:2]=='(一' or line1[:2]=='(二' or line1[:2]=='(三' or line1[:2]=='(四' or line1[:2]=='(五'  or line1[:2]=='(六'  or line1[:2]=='(七' or line1[:2]=='(八' or line1[:2]=='(九' or line1[:2]=='(十'  :      
                           list1_dg.append(line1)
                    elif   line1[:2]=='1.' or line1[:2]=='2.' or line1[:2]=='3.' or line1[:2]=='4.' or line1[:2]=='5.'  or line1[:2]=='6.'  or line1[:2]=='7.' or line1[:2]=='8.' or line1[:2]=='9.' or line1[:3]=='10.'  :
                           list1_dg.append(line1)    
    ct=ct.replace(' ', '')
    sg=''
    if len(list1_dg)>0:
        for t in range(len(list1_dg)):
            sg=sg+'，'+list1_dg[t]

    return (sg,ct,len(list1_dg))


from transformers import BertTokenizer,TFBertModel
from sklearn.metrics.pairwise import cosine_similarity  # 余弦距离
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\LJM\PycharmProjects\pythonProject5\中文阅读理解\bert-base-chinese')
model=TFBertModel.from_pretrained(r'C:\Users\LJM\PycharmProjects\pythonProject5\中文阅读理解\bert-base-chinese')
def cos_similar(text1,text2):
    in1=tokenizer(text1,padding=True,truncation=True, max_length=500,return_tensors='tf')
    in2=tokenizer(text2,padding=True,truncation=True, max_length=500,return_tensors='tf')
    out1=model(in1)
    bert_cls_hidden_state1 = out1[0][:,0,:] 
    v1=np.array(bert_cls_hidden_state1[0])
    out2=model(in2)
    bert_cls_hidden_state2 = out2[0][:,0,:] 
    v2=np.array(bert_cls_hidden_state2[0])
    cos=cosine_similarity([v1, v2])[0][1]
    return cos


# 设置全局属性
st.set_page_config(
    page_title='页面标题',
    page_icon='☆☆☆ ',
    layout='wide'
)

st.title('基于Bert的竞赛论文辅助评阅系统')# 设置页面标题


with st.form("参数输入"):
    st.header("输入：")
    # 定义输入参数框
    st.subheader('论文上传数量：（至少上传10篇以上）')
    num = st.number_input('请输入上传论文数量:', min_value=0, max_value=500)
    st.subheader('输入评分权重：（最终得分=评分要点得分*权重+摘要质量得分*权重+写作水平得分*权重）')
    pfyd_text = st.number_input("请输入评分要点所占权重（小数）:",  format="%.2f")
    zy_text = st.number_input("请输入摘要所占权重（小数）:",  format="%.2f")
    xzsp_text = st.number_input("请输入写作水平所占权重（小数）:", format="%.2f")
    st.subheader('输入综评得分占比：（1~10分，注意：“综评”不等于上文提到的“最终得分”，上文提到的“最终得分”为小数')
    bl810_text = st.number_input("请输入8~10分所占比例不超过（小数）:", format="%.2f")
    bl67_text = st.number_input("请输入6~7分所占比例不超过（小数）:", format="%.2f")
    bl45_text = st.number_input("请输入4~5分所占比例不超过（小数）:",  format="%.2f")
    
    st.subheader('上传评分要点文件，格式如示例：')
    #输入文件要求
    A = st.file_uploader('上传评分要点以及要求（如图所示详细内容放第二列，评分类型放第一列，只支持excle表格）', type=['xls','xlsx'])
    if A is not None:
      A=pd.read_excel(A)   
    else:
        st.write('')
    yq=pd.read_excel(r'C:\Users\LJM\PycharmProjects\pythonProject5\中文阅读理解\2024.bert学习\评分要点拆解.xlsx')
    st.write(yq)
    # 创建文件上传组件
    st.subheader('上传论文pdf文件：')
    uploaded_files = st.file_uploader('上传需要评阅的论文:(实际上传论文数量必须与输入的一致)', accept_multiple_files=True,type=['pdf'])
# 当表单被提交时，继续运行后续代码
    submitted = st.form_submit_button("提交并开始辅助评阅打分")



#

if submitted:
    if uploaded_files is not None:  
        res_data=pd.DataFrame()          
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read()
            # 将文件内容写入到一个临时文件中
            with open('temp.pdf', 'wb') as f:
                f.write(file_content)
                    
                with open('temp.pdf', 'rb') as file:
                    
                    reader=PyPDF2.PdfReader(file)#一个一个遍历传输进来的文件,haiyao转换成二进制文件
                   
                    
                    list_table=[]
                    ima_n=0
                    biao_n=0
                    index_s=[]
                    fname=uploaded_file.name         
                    
                    #总文件页数
                    num_pages = len(reader.pages)
                    
                    #获取文件图片个数
                    try:
                        for  page in reader.pages:
                            for im in page.images:
                                ima_n=ima_n+1
                    except:
                        print(fname+'图片有损坏')
                    
                    #获取文件表格个数
                    reader = pdfplumber.open(file)
                    for  page in reader.pages:
                         tables = page.extract_tables()
                         list_table.extend(tables)
                    biao_n=len(list_table)
                    
                    #调用自定义函数，获取论文大纲字符串、论文正文、大纲条数
                    r=dg_get(fname)
                    context=r[1]#获取所有内容
                    
                    #获取论文摘要
                    index1=context.find('摘要',0,len(context))
                    index2=context.find('关键词',0,len(context))
                    abstract=context[index1+2:index2]
                    
                    #获取正文中的纯中文字符
                    res1=''.join(re.findall('[\u4e00-\u9fa5]',context))
                    
                    #提取正文摘要，并基于提取的正文摘要，与原论文摘要，计算相似度
                    if len(context)>2 and len(abstract)>2:
                        tr4s = TextRank4Sentence()
                        tr4s.analyze(text=context, lower=True, source ='all_filters')
                        nn1=[]
                        for item in tr4s.get_key_sentences(num=1):
                            nn1.append(item.sentence)
                            #提取的摘要：正文
                        cossim=cos_similar(nn1[0], abstract)#提取的摘要和原文摘要计算相似度
                        cossz1=cos_similar(A.iloc[0,1], nn1[0])#提取的摘要与评分要点的相似度（完整性）
    
                    else:
                        cossim=-1
                        cossz1=-1
    
                    
                    #计算摘要与评分要点的相似度（完整性，实质性）
                    if len(abstract)>2:
                        coswz=cos_similar(A.iloc[0,1], abstract) #完整性
    
                    else:
                        coswz=-1  
    
                    
                    #评分要点与大纲相似度
                    if len(r[0])>2:
                        cossd1=cos_similar(A.iloc[0,1], r[0]) #完整性
    
                    else:
                        cossd1=-1  
                    
                    if len(r[0])>2 and len(abstract)>2:
                        cosdz=cos_similar(r[0], abstract)
                    else:
                        cosdz=-1
                        
                    res_dict={
                    '文件名':fname,
                    '页数':num_pages,
                    '总字符数':len(context),
                    '中文字符数':len(res1),
                    '图片个数':ima_n,
                    '表格个数':biao_n,
                    '大纲条数':r[2],
                    '正文与摘要相似度':cossim,
                    '大纲与摘要相似度':cosdz, #x7
                    '评分要点与摘要相似度':coswz,
                    '评分要点与大纲相似度':cossd1,
                    '评分要点与正文相似度':cossz1
    
    
                        }
                    tttt = pd.DataFrame(pd.Series(res_dict)).T 
                    res_data=pd.concat([res_data,tttt])
                        
                    
    #===================================评分===================================                          
        st.write(res_data)                
        import pandas as pd
        dt1=res_data
        #更改变量名
        col2=list(dt1['文件名'])
        dt2=dt1
        dt2.index=col2
        dt2=dt2.drop('文件名',axis=1)
        dt2.columns=['X4','X5','X6','X7','X8','X9','X10','X11','X1','X2','X3']
        #==================无错误================================
        #将数据标准化
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()
        dt2=scaler.fit_transform(dt2)
        dt2=pd.DataFrame(dt2)
        dt2.columns=['X4','X5','X6','X7','X8','X9','X10','X11','X1','X2','X3']
        #各类指标提取合并
        #1.评分要点评价：指标包括评分要点（完整性）与正文的相似度（X1）、评分要点（完整性）与摘要的相似度（X2）、评分要点（完整性）与大纲的相似度（X3）。
        X_wanzheng=dt2[['X1','X2','X3']]
        #2.摘要质量：指标包括摘要与正文的相似度（X7）、摘要与大纲的相似度（X8）
        X_zhaiyao=dt2[['X10','X11']]
        #3.写作水平：指标包括页数（X4）、总字数（X5）、中文字符数（X6）、图片个数（X7）、表格个数（X8），大纲条数（X9）以及以上涉及到的指标X1~X3，共11个。
        X_level=dt2
        st.write(X_wanzheng)
        
        
        
        #法一：逐一主成分分析，排序出结果
        #进行主成分分析
        from sklearn.decomposition import PCA
        #评分要点主成分分析
        pca=PCA(n_components=1)
        X_pca_wzx=pca.fit_transform(X_wanzheng).flatten()
        #摘要质量主成分分析
        X_pca_zy=pca.fit_transform(X_zhaiyao).flatten()  
        #写作水平主成分分析
        X_pca_xzsp=pca.fit_transform(X_level).flatten()
        
        
        x1={'x':X_pca_wzx,'tem':range(len(X_pca_wzx))}
        x2={'x':X_pca_zy,'tem':range(len(X_pca_zy))}    
        x3={'x':X_pca_xzsp,'tem':range(len(X_pca_xzsp))} 
        a1=pd.DataFrame(x1)    
        a2=pd.DataFrame(x2)
        a3=pd.DataFrame(x3)
        
     

  
        # 生成列表  
        def generate_uniform_distribution_integers(n):
            # Create a list to store the elements
            result = []
            step = 10 / n
            for i in range(n):
                result.append(int((i * step) + 1))
            return result


        numbers=generate_uniform_distribution_integers(num)
  
        # 打印生成的列表  
        def sort_score(a):
            te=a.sort_values(by='x',ascending=True)
            te['rank']=numbers
            te=te.sort_values(by='tem',ascending=True)
            rank1=te['rank']
            return rank1
        
        score_1=sort_score(a1)
        score_2=sort_score(a2)
        score_3=sort_score(a3) 

        # 评分要点权重40%  
        weight_pfyd = pfyd_text  
        #摘要质量权重20%
        weight_zy=zy_text  
        #写作水平权重
        weight_xzsp =xzsp_text
        #计算最终的得分公式
        res_1=weight_pfyd *score_1 + weight_zy*score_2 + weight_xzsp*score_3
        

        #输入得分比例本文8-10分占比不超过3%，6-7分比例不超过12%，4-5分比例不超过20%。
        #每一组的结束位置
        n1_8_10=round(num*bl810_text)
        n1_6_7=round(num*bl67_text)+n1_8_10
        n1_4_5=round(num*bl45_text)+n1_6_7
        n1_1_3=num
        #每一组的个数
        n_8_10=round(num*bl810_text)
        n_6_7=round(num*bl67_text)
        n_4_5=round(num*bl45_text)
        n_1_3=num-n1_8_10-n1_4_5-n1_6_7
        rk=[]
        for i in range(1,num+1):
            if i<=n1_8_10:#0.03
                if i<round(n_8_10*0.2):
                    rk.append(10)
                    continue
                elif round(n_8_10*0.2)<=i<round(n_8_10*0.5):
                    rk.append(9)
                    continue
                else:
                    rk.append(8)
                    continue
         
            elif i<=n1_6_7:
                if i<round(n1_1_3*0.08):
                    rk.append(7)
                    continue
                else: 
                    rk.append(6)
                    continue
        
            elif i<=n1_4_5:
                if i<round(n1_1_3*0.17):
                    rk.append(5)
                    continue
                else: 
                    rk.append(4)
                    continue
        
            else:  
                if i<round(n1_1_3*0.5):
                    rk.append(3)
                    continue
                elif round(n1_1_3*0.5<=i<n1_1_3*0.7):
                    rk.append(2)
                    continue
                else:
                    rk.append(1)
                    continue
        
        res_1=pd.DataFrame(res_1)  
        res_1['index_tem']=range(len(res_1)) 
        re=res_1.sort_values(by='rank',ascending=False)
        re['y']=rk
        re=re.sort_values(by='index_tem',ascending=True)
        
        
        
        result_1={'论文编号':col2,'评分要点':score_1,'摘要':score_2,'写作水平':score_3,'得分':re['y']}
        result_1=pd.DataFrame(result_1)
        st.write(result_1)
        
        sc_10=len(result_1.loc[result_1['得分']==10,:])
        sc_9=len(result_1.loc[result_1['得分']==9,:])
        sc_8=len(result_1.loc[result_1['得分']==8,:])
        sc_7=len(result_1.loc[result_1['得分']==7,:])
        sc_6=len(result_1.loc[result_1['得分']==6,:])
        sc_5=len(result_1.loc[result_1['得分']==5,:])
        sc_4=len(result_1.loc[result_1['得分']==4,:])
        print('综合得分为10分的有：',sc_10,'个')
        print('综合得分为9分的有：',sc_9,'个')
        print('综合得分为8分的有：',sc_8,'个')
        print('综合得分为7分的有：',sc_7,'个')
        print('综合得分为6分的有：',sc_6,'个')
        print('综合得分为5分的有：',sc_5,'个')
        print('综合得分为10分的有：',sc_4,'个')
        
        print("8-10分所占比例:",(sc_10+sc_9+sc_8)/len(result_1),'<=0.03。故符合比例')
        print("6-7分所占比例:",(sc_7+sc_6)/len(result_1),'>=0.1。故符合比例')
        print("4-5分所占比例:",(sc_4+sc_5)/len(result_1),'>=0.2。故符合比例')
        print("6-10分所占比例:",(sc_10+sc_9+sc_8+sc_7+sc_6)/len(result_1),'<=0.15。故符合比例')
        print("4-10分所占比例:",(sc_10+sc_9+sc_8+sc_7+sc_6+sc_4+sc_5)/len(result_1),'<=0.35。故符合比例')
        
        from pandas import ExcelWriter 
        
        excel_file_name = 'output.xlsx'
        
        # 使用pandas ExcelWriter和XlsxWriter引擎创建Excel文件
        with ExcelWriter(excel_file_name, engine='xlsxwriter') as writer:
             result_1.to_excel(writer, index=False, sheet_name='Sheet1')
            # 如果需要，可以添加更多的sheet或者格式化
            # ...
        
        # 使用Streamlit的file_uploader组件来提供下载链接
        with open(excel_file_name, "rb") as file:
             btn = st.download_button(
                label="Download Excel file",
                data=file,
                file_name=excel_file_name,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    else:
        st.text('未上传Excel文件或文件格式不正确。')
        
    
        


    
    





