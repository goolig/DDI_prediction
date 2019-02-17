import os

from d2d_DAL import drug_data_reader
from d2d_preprocess import drugs_preproc
from dataset_dates import d2d_versions_metadata
from utils import unpickle_object, pickle_object


class d2d_releases_reader:

    relase_base_path = os.path.join('pickles','data','DrugBank_releases')
    release_base_file = os.path.join(relase_base_path ,'%s', 'drugbank_all_full_database.xml.zip')
    file_reader_pickle_path = os.path.join('pickles','file_reader')
    preproc_pickle_path = os.path.join('pickles','preproc')
    pickle_extension = '.pickle'

    def __init__(self):
        pass

    def read_and_preproc_release(self, version ='', force_read_file=False):
        self.force_read=force_read_file
        drug_reader = self.read_release(version)
        preproc = self.preproc_release(drug_reader, version)
        return drug_reader,preproc

    def normalize_version(self, version):
        if version == '':
            version = d2d_versions_metadata[0]['VERSION']
            print('using latest version')
        elif version == '-1':
            version = d2d_versions_metadata[-1]['VERSION']
            print('using oldest version')
        print("Release number:", version)
        release_metada = self.get_release_metadata(version)
        print('relese metadata:', release_metada)
        return version

    def read_release(self, version):
        print('reading release...')
        version = self.normalize_version(version)
        file_reader_pickle_path = self.get_file_reaeder_pickle_for_version(version)
        try:
            if self.force_read:
                raise ValueError('Forcing read')
            drug_reader = unpickle_object(file_reader_pickle_path)
        except:
            print('failed to unpickle')
            release_path = self.get_relese_path(version)
            drug_reader = drug_data_reader(release_path)
            drug_reader.read_data_from_file()
            pickle_object(file_reader_pickle_path, drug_reader)
        # drug_id_to_name = drug_reader.drug_id_to_name
        return drug_reader

    def preproc_release(self, drug_reader, version):

        print('postprocessing release...')
        version = self.normalize_version(version)
        preproc_pickle_path = self.get_preproc_pickle_for_version(version)
        try:
            if self.force_read:
                raise ValueError('Forcing read')
            preproc = unpickle_object(preproc_pickle_path)
        except:
            print('failed to unpickle')
            print('num all drugs in reader:', len(drug_reader.all_drugs))
            preproc = drugs_preproc(drug_reader.drug_to_interactions, drug_reader.all_drugs)
            preproc.calc_valid_drugs_print_summary()
            preproc.create_valid_drug_interactions()
            pickle_object(preproc_pickle_path, preproc)
        return preproc

    def get_release_metadata(self, version):

        for x in d2d_versions_metadata:
            if x['VERSION'] == version:
                return x
        assert False,'cant find version %s in metadata array' % version

    def get_file_reaeder_pickle_for_version(self, version):
        return d2d_releases_reader.file_reader_pickle_path + version + d2d_releases_reader.pickle_extension

    def get_preproc_pickle_for_version(self, version):
        return d2d_releases_reader.preproc_pickle_path + version + d2d_releases_reader.pickle_extension

    def get_relese_path(self, version):
        return d2d_releases_reader.release_base_file % version
