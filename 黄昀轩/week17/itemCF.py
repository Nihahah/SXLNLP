import openpyxl
import torch.nn as nn
import torch
import math
import numpy as np
from collections import defaultdict


# 将数据格式转化为self_item的打分矩阵格式
# 对应电影id和其名字
def build_u2i_matrix(item_name_data_path, user_item_score_data_path, write_file=False):
	item_id_to_item_name = {}
	with open(item_name_data_path, encoding="ISO-8859-1") as f:
		for line in f:
			item_id = int(line.strip().split("|")[0])
			item_name = line.strip().split("|")[1]
			item_id_to_item_name[item_id] = item_name
	total_movie_count = len(item_id_to_item_name)
	print('加载了%d个电影条目' % total_movie_count)
	# return item_id_to_item_name
	# 用户打分
	user_to_rating = {}
	with open(user_item_score_data_path, encoding='ISO-8859-1') as f:
		for line in f:
			user_id, item_id, score, time_stamp = line.split("\t")
			user_id, item_id, score = int(user_id), int(item_id), int(score)
			if user_id not in user_to_rating:
				user_to_rating[user_id] = [0] * total_movie_count
				user_to_rating[user_id][item_id - 1] = score
	print("total user:", len(user_to_rating))
	# print(user_to_rating[244])
	
	if not write_file:
		return user_to_rating, item_id_to_item_name
	
	# 写入excel便于查看
	workbook = openpyxl.Workbook()
	sheet = workbook.create_sheet(index=0)
	# 第一行：user_id, movie1, movie2...
	header = ["user_id"] + [item_id_to_item_name[i + 1] for i in range(total_movie_count)]
	sheet.append(header)
	for i in range(len(user_to_rating)):
		# 每行：user_id, rate1, rate2...
		line = [i + 1] + user_to_rating[i + 1]
		sheet.append(line)
	workbook.save("user_movie_rating.xlsx")
	return user_to_rating, item_id_to_item_name


# 输入np.array的高维数组
def cosine_distance(vector1, vector2):
	ab = np.dot(vector1, vector2)
	a1 = np.sqrt(np.sum(np.square(vector1)))
	a2 = np.sqrt(np.sum(np.square(vector2)))
	# 余弦距离越大越接近
	return ab / (a1 * a2)


# 输出记录与当前电影相似度从高到底的排序  {movie_a:[[movie_b,1],[movie_c,0.99],...]}
def find_similar_item(user_to_rating):
	# 记录每个电影对应所有用户对它的打分
	items_to_vector = {}
	total_users = len(user_to_rating)
	print('length:%d' % len(user_to_rating))
	for user, user_rating in user_to_rating.items():
		for movie_id, score in enumerate(user_to_rating):
			movie_id += 1
			if movie_id not in items_to_vector:
				items_to_vector[movie_id] = [0] * (total_users + 1)
			items_to_vector[movie_id][user] = score
	# 记录每个电影与其他电影的相似度（从高到低排序）
	return find_similar_user(items_to_vector)


# 输出记录与当前索引用户相似度从高到底的排序  {user_a:[[user_b,1],[user_c,0.99],...]}
def find_similar_user(user_to_rating):
	user_to_similar_user = {}
	score_buffer = {}
	for user_a, rating_a in user_to_rating.items():
		similar_user = []
		for user_b, rating_b in user_to_rating.items():
			if user_a == user_b or user_a >= 100 or user_b >= 100:
				continue
			if '%d_%d' % (user_a, user_b) in score_buffer:
				similarity = score_buffer['%d_%d' % (user_b, user_a)]
			else:
				similarity = cosine_distance(np.array(rating_a), np.array(rating_b))
				score_buffer['%d_%d' % (user_a, user_b)] = similarity
			similar_user.append([user_b, similarity])
		similar_user = sorted(similar_user, reverse=True, key=lambda x: x[1])
		user_to_similar_user[user_a] = similar_user
	return user_to_similar_user


def item_cf(user_id, item_id, simi_item, user_to_rating, topn=10):
	pred_score = 0
	count = 0
	print(item_id)
	
	for item_h, similarity in simi_item[item_id][:10]:
		# 用户评分过的物品
		for item_id, score in enumerate(user_to_rating[user_id]):
			if score == 0:
				continue
			else:
				# item_h再相似物品集中对应的指定物品相似度
				simi_item_resorted = sorted(simi_item[item_h], reverse=True, key=lambda x: x[0])
				simi_score = simi_item_resorted[item_id + 1][1]
				# print(simi_score)
				pred_score += float(simi_score * score)
				count += 1
			# print(pred_score)
	pred_score /= count + 1e-5
	return pred_score


def user_cf(user_id, item_id, user_to_similar_user, user_to_rating, topn=10):
	pred_score = 0
	count = 0
	for similar_user, similarity in user_to_similar_user[user_id][:topn]:
		# 相似用户对这部电影的打分
		rating_by_similiar_user = user_to_rating[similar_user][item_id - 1]
		# 分数*用户相似度，作为一种对分数的加权，越相似的用户评分越重要
		pred_score += rating_by_similiar_user * similarity
		# 如果这个相似用户没看过，就不计算在总数内
		if rating_by_similiar_user != 0:
			count += 1
	pred_score /= count + 1e-5
	return pred_score


def movie_recommend(user_id, similar_user, similar_items, user_to_rating, item_to_name, topn=10):
	# 候选集就是用户没看过的电影
	unseen_items = [item_id + 1 for item_id, rating in enumerate(user_to_rating[user_id]) if rating == 0]  # 主义这个条件的写法
	res = []
	for item_id in unseen_items:
		score = item_cf(user_id, item_id, similar_items, user_to_rating)
		res.append([item_to_name[item_id], score])
	res = sorted(res, reverse=True, key=lambda x: x[1])
	return res[:topn]
	
	pass


if __name__ == '__main__':
	user_item_score_data_path = 'ml-100k/u.data'
	item_name_data_path = 'ml-100k/u.item'
	# user_to_rating,item_to_name = build_u2i_matrix(user_item_score_data_path,item_name_data_path)
	
	user_to_rating, item_to_name = build_u2i_matrix(item_name_data_path, user_item_score_data_path)
	
	simi_item = find_similar_item(user_to_rating)
	simi_user = find_similar_item(user_to_rating)
	
	#
	# print(user_to_rating[1])
	# print(simi_item)
	# print(sorted(simi_item,reverse=True,key = lambda x:x[0]))
	
	while True:
		user_id = int(input("输入用户id："))
		recommends = movie_recommend(user_id, simi_user, simi_item, user_to_rating, item_to_name)
		for name, score in recommends:
			print(name, score)

# similar_user = find_similar_user(user_to_rating)
# similar_item = find_similar_item(user_to_rating)
