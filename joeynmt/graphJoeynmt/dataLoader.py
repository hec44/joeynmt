from torchtext import data
from torchtext.data import Dataset, Iterator, Field
import numpy as np
from tqdm import tqdm
import pdb
class GraphTranslationDataset(data.Dataset):
    """Defines a dataset for machine translation with a graph reprsentation on the input and levi graph transformations."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, src_file, trg_file, fields, **kwargs):
        """Create a TranslationDataset given paths and fields.
        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """
        if not isinstance(fields[0], (tuple, list)):
            fields = [('src', fields[0]), ('trg', fields[1]),\
                     ('edge_org', fields[2]), ('edge_trg', fields[3]), ('positional_en',fields[4])]

        examples = []
        source_words,origins,targets=self.read_conllu(src_file)
        target_words=self.read_text_file(trg_file)
        target_words,source_words,origins,targets,pes\
           =self.gen_pes(target_words,source_words,origins,targets)
        
        if len(source_words) != len(target_words):
          target_words=target_words[:-1]
        assert len(source_words)==len(target_words),"Mismatch of source and tagret sentences"
        for i in range(len(source_words)):
                src_line, trg_line = " ".join(source_words[i]),target_words[i]
                src_line, trg_line = src_line.strip(), trg_line.strip()
                
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line," ".join(origins[i])," ".join(targets[i]),\
                        " ".join(pes[i])],\
                        fields))
        super(GraphTranslationDataset, self).__init__(examples, fields, **kwargs)

    @classmethod
    def splits(cls, exts, fields, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.
        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, fields, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, fields, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, fields, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)
    def read_conllu(self,path):
        """
        creates three lists: one with the sentences, and two that represent the edges fot he graph
        Argmunets:
            path: path to a file with sentences in the ConLL-U standard
        """
        f=open(path,'r')
        lines=f.readlines()
        f.close()
        words=[]
        origins=[]
        targets=[]
        edges=[]
        temp_words=[]
        temp_origins=[]
        temp_targets=[]
        temp_edges=[]
        for line in lines:
            if line=='\n'or line=='':
                words.append(temp_words)
                origins.append(np.array(temp_origins))
                targets.append(np.array(temp_targets))
                edges.append(temp_edges)
        
                temp_words=[]
                temp_origins=[]
                temp_targets=[]
                temp_edges=[]
    
            else:
                splits=line.split('\t')
                temp_words.append(splits[1])
                temp_origins.append(int(splits[0]))
                temp_targets.append(int(splits[6]))
                temp_edges.append("<"+splits[7]+">")
        for i in range(len(words)):
            new_origins=origins[i]-1
            edges_positions=np.arange(len(words[i]),2*len(words[i]))
            new_targets=edges_positions.copy()
            
            edge_targets=targets[i]-1
            root_pos=np.argmin(edge_targets)
            edge_targets = np.delete(edge_targets, [root_pos])
            edge_origins = np.delete(edges_positions,[root_pos])
            origins[i] = [str(num) for num in list(np.concatenate((new_origins,edge_origins)))]
            targets[i] = [str(num) for num in list(np.concatenate((new_targets,edge_targets)))]
            assert len(targets[i])==len(origins[i])
            words[i]=words[i]+edges[i]
            
        return words,origins,targets
    
    def read_text_file(self,path):
        """Read a text file 
        Argmunets:
            path: path to a normal txt file
        """
        f=open(path,'r')
        lines=f.readlines()
        f.close()
        return lines
    def gen_pe(self,words,org,trg,root_kw="<root>"):
        """Calculates the min distance to the root to each node using BFS
        Argmunets:
            words: all the words of the sentence
            org: a list with the origin of each edge
            trg: a list with the target of each edge
            root_kw: the keyword of the roo tag in the sentence
        """
        start=None
        for ind,word in enumerate(words):
            if word==root_kw:
                start=ind
                continue
        assert start!=None,"sentence does not have a <root> tag"
        visited=[start]
        distance_queue=[1]
        distances=['0']*len(words)
        while len(visited)!=0:
            for index,node in enumerate(trg):
                if str(node)==str(visited[0]):
                    distances[int(org[index])]=str(distance_queue[0])
                    visited.append(org[index])
                    distance_queue.append(distance_queue[0]+1)
            visited.pop(0)
            distance_queue.pop(0)
        return distances
    def gen_pes(self,target_words,source_words,orgs,trgs,root_kw="<root>"):
        """
        Generates the positional embeddings for all the sentences in the dataset
        argmunets:
            source_words: a list of sentences 
            orgs: a list of lists of ede origins
            trgs: a list of lists of the edge targets
            root_kw:the keyword of the root tag in the senteces
        """
        pes=[]
        banned_is=[]
        for i in tqdm(range(len(source_words))):
            if "-1" in " ".join(trgs[i]):
              banned_is.append(i)
            else:
              pes.append(self.gen_pe(source_words[i],orgs[i],trgs[i],root_kw))
        for i in tqdm(banned_is):
          del source_words[i]
          del orgs[i]
          del trgs[i]
          del target_words[i]

        return target_words,source_words,orgs,trgs,pes
