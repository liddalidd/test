from pyspark import SparkConf, SparkContext
import warnings
warnings.filterwarnings('ignore')

from pyspark.sql import SparkSession

import argparse
from pyspark.ml.feature import Word2Vec
from pyspark.sql.functions import split,posexplode
from pyspark.ml.feature import HashingTF,IDF 
from pyspark.ml.recommendation import ALS   

# Cosine similarity function
def cos_sim(x,y):
    return x.dot(y)/(x.norm(2)*y.norm(2))

def getdata(data):
    # select (user_id, replyto_id)
    user_rp = data.select('user_id', 'replyto_id')\
                    .where("replyto_id is not null")\
                    .distinct()

    # select (user_id, retweet_id )
    user_rt = data.select('user_id', 'retweet_id')\
                    .where("retweet_id is not null")\
                    .distinct()

    # concatenate two dataframes as a new dataframe (uid, tid)
    user_rp_rt = user_rp.union(user_rt).toDF("uid", "tid").cache()
    return user_rp_rt

def TFIDFRes(data):
    # sc.broadcast(user_rp_rt)
    user_rp_rt = getdata(data)
    # get document representation for each user
    user_doc = user_rp_rt.rdd.map(lambda row: (row[0], str(row[1])))\
                                  .groupByKey()\
                                  .map(lambda row: (row[0], [e for e in row[1]])) \
                                  .toDF(["uid", "Document"])

    # IDF
    hashingTF = HashingTF(inputCol = "Document", outputCol = "TF")
    hashingTF.setNumFeatures(10)
    idf = IDF(inputCol = "TF", outputCol = "Vec")

    user_doc_tf = hashingTF.transform(user_doc)

    # get user vector
    user_vec = idf.fit(user_doc_tf).transform(user_doc_tf)\
                                    .drop("Document")\
                                    .drop("TF")

    # get target user vector
    target_user_vec = user_vec.filter("uid = "+ str(USER_ID)).collect()[0][1]

    # get other user vectors
    other_user_vec_rdd = user_vec.filter("uid != "+ str(USER_ID)).rdd\
                            .map(lambda row: (row[0],row[1]))

    # calculate the similarity between each user to the target user
    user_sim = other_user_vec_rdd.map(lambda row: (row[0], cos_sim(target_user_vec,row[1])))

    # Find the top 5 users
    top_users = user_sim.takeOrdered(5, lambda row:-row[1])
    return top_users
 

 
## Word2Vec
def Word2VecRes(data):
    tweet_text = data.select('id',split("text"," ").alias('words')).cache()
    word2vec = Word2Vec(vectorSize = 10, inputCol="words", outputCol="model")
    model  = word2vec.fit(tweet_text)
    word_vecs = model.getVectors()

    # document vector
    doc_vec = tweet_text.select('id', posexplode("words").alias("pos", "word"))\
                        .join(word_vecs,'word').select('id','vector')\
                        .rdd.map(lambda row:(row[0],row[1]))\
                        .reduceByKey(lambda x,y:x+y)

    user_rp_rt = getdata(data)
    # user vector
    user_vec = user_rp_rt.rdd.map(lambda row:(row[1],row[0]))\
                            .join(doc_vec)\
                            .map(lambda row:(row[1][0],row[1][1]))\
                            .reduceByKey(lambda x,y:x+y)

    # get target user vector
    target_user_vec = user_vec.filter(lambda user_vec_pair:user_vec_pair[0] == USER_ID).collect()[0][1]

    # get other user vectors
    other_user_vec_rdd = user_vec.filter(lambda user_vec_pair:user_vec_pair[0] != USER_ID)

    # calculate the similarity between each user to the target user
    user_sim = other_user_vec_rdd.map(lambda row: (row[0], cos_sim(target_user_vec,row[1])))

    # Find top 5 users
    top_users = user_sim.takeOrdered(5, lambda row:-row[1])
    return top_users

def getDataDict(data):
    # user_id collection
    user_ids = data.select("user_id").distinct()\
                .rdd.map(lambda row:row[0]).collect()

    # user_id dictionary
    user_id_dic = {}
    for u in user_ids:
        user_id_dic[u] = len(user_id_dic)

    # user_mentions collection
    men_ids = data.select('user_mentions')\
                    .filter("user_mentions is not null")\
                    .rdd.flatMap(lambda row:[e[0] for e in row[0]])\
                    .distinct().collect()

    # user_mentions dictionary
    men_id_dic = {}
    for m in men_ids:
        men_id_dic[m] = len(men_id_dic)
        

    return user_ids, user_id_dic, men_ids, men_id_dic

def RecommandRes(data):

    # Broadcast user_id_dic and men_id_dic 
    # sc.broadcast(user_id_dic)
    # sc.broadcast(men_id_dic)

    user_ids, user_id_dic, men_ids, men_id_dic = getDataDict(data)
    # user_id, user_mentions_id, count
    user_men_rdd = data.select("user_id","user_mentions")\
                        .where("user_mentions is not null")\
                        .rdd.map(lambda row:(user_id_dic[row[0]],row[1]))\
                        .flatMap(lambda row:[(row[0], men_id_dic[men[0]], 1) for men in row[1]])


    user_men_rate = user_men_rdd.map(lambda row:((row[0],row[1]),row[2]))\
                            .reduceByKey(lambda x,y:x+y)\
                            .map(lambda row:(row[0][0], row[0][1], row[1]))\
                            .toDF(["uid", "mid", "rating"])


    # Collaborative Filtering
    als = ALS(userCol="uid", itemCol="mid", ratingCol='rating',
              coldStartStrategy="drop")

    model = als.fit(user_men_rate)
     
    # Generate top 5 recommendations for each user
    rec_res = model.recommendForAllUsers(5).collect()

    return rec_res

def getOutputFile(res1, res2, res3):
    res=['Workload1','TFIDF:']
    for i in res1:
        res.append(i[0])
    res.append(" ")
    res.append("Word2Vec:")
    for j in res2:
        res.append(j[0])

    user_ids, user_id_dic, men_ids, men_id_dic = getDataDict(data)
    res.append(" ")
    res.append('Workload2:')
    res.append("Top 5 recommendations for each user")
    for rec_row in res3:
        res.append(user_ids[rec_row[0]])
        mrec = []
        for rec in rec_row[1]:
            mrec.append(men_ids[rec[0]])
        res.append(mrec)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", help="the output path") 
    args = parser.parse_args()
    output_path = args.output 

    spark = SparkSession \
            .builder \
            .appName("Python Spark SQL basic example") \
            .getOrCreate()

    spark_conf = SparkConf().setAppName("Assignment2")
    sc = SparkContext.getOrCreate(spark_conf)

    path = './tweets.json'

    data = spark.read.option("multiline", "true").json(path)

    USER_ID = 1340169345914318848

    tfidfRes = TFIDFRes(data)
    w2vRes = Word2VecRes(data)
    recRes = RecommandRes(data)
    res = getOutputFile(tfidfRes, w2vRes, recRes) 

    sc.parallelize(res).coalesce(1, shuffle = False).saveAsTextFile(output_path)





























