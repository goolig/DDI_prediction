import itertools
import pickle


def array_to_dict(array):

    return dict([(v, i) for i, v in enumerate(array)])

def flatten_list(l):
    #return [item for sublist in l for item in sublist]
    return list(itertools.chain.from_iterable(l))

def pickle_object(path,object):
    print('pickling:',path)
    filehandler = open(path, "wb")
    pickle.dump(object, filehandler)
    filehandler.close()
    print('done pickling:', path)

def unpickle_object(path):
    print('trying to unpick:', path)
    file = open(path, 'rb')
    object_file = pickle.load(file)
    file.close()
    print('done unpickling:', path)
    return object_file

def print_array_file(array,file_path):
    thefile = open(file_path, 'w')
    for item in array:
        thefile.write("%s\n" % ", ".join([str(x) for x in item]))

def make_mat_sym(m,func):
    for i in range(m.shape[0]):
        for j in range(i + 1, m.shape[0]):
            m[i, j] = func(m[i, j], m[j, i])
            m[j, i] = m[i, j]