from janome.dic import UserDictionary
from janome import sysdic
user_dict = UserDictionary('res/neologd.csv', 'utf8', 'ipadic', sysdic.connections)
user_dict.save('neologd')