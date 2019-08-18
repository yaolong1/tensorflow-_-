import pymysql
coon = pymysql.connect(
    host='127.0.0.1',
    user='root',
    passwd='yin7372175240000',
    port=3306,
    db='ssm',
    charset = 'utf8'
    #port必须写int类型
    #charset必须写utf8，不能写utf-8
)
cur = coon.cursor()  #建立游标
# cur.execute("select * from account")  #查询数据
cur.execute('insert into account(name,money) VALUE ("pzp",2000);')
coon.commit()
res = cur.fetchall()    #获取结果
cur.close()     #关闭游标
coon.close()    #关闭连接
# ---------------------------------------------------------------------------
# #如果是插入数据，则要commit一下，把第9行换成以下两行
# cur.execute('insert into stu(name,sex) VALUE ("pzp","man");')
# coon.commit()