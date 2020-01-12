from fakenews_detector.preprocessor_twitter import PreProcessorTwitter
from fakenews_detector.preprocessor_texts import PreProcessorTexts


def setup():
    print("Setting up fakenewsdata1_randomPolitics")
    p = PreProcessorTexts(language='en', embeddings_path='embeddings/en/model.bin', platform='Websites', dataset="fakenewsdata1_randomPolitics")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="fakenewsdata1_randomPolitics", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="fakenewsdata1_randomPolitics", class_label="True")
    print("--------------------------")
    print()

    print("Setting up Bhattacharjee")
    p = PreProcessorTexts(language='en', embeddings_path='embeddings/en/model.bin', platform='Websites', dataset="Bhattacharjee")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="Bhattacharjee", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="Bhattacharjee", class_label="True")
    print("--------------------------")
    print()
    
    print("Setting up btv-lifestyle")
    p = PreProcessorTexts(language='bg', embeddings_path='embeddings/bg/model.txt', platform='Websites', dataset="btv-lifestyle")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="btv-lifestyle", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="btv-lifestyle", class_label="True")
    print("--------------------------")
    print()

    print("Setting up FakeBrCorpus")
    p = PreProcessorTexts(language='pt', embeddings_path='embeddings/pt/model.txt', platform='Websites', dataset="FakeBrCorpus")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="FakeBrCorpus", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Websites", dataset="FakeBrCorpus", class_label="True")
    print("--------------------------")
    print()

    print("Setting up tweets_br")
    p = PreProcessorTwitter(language='pt', embeddings_path='embeddings/pt/model.txt',  dataset="tweets_br")
    p.convert_rawdataset_to_dataset(platform="Twitter", dataset="tweets_br", class_label="False")
    p.convert_rawdataset_to_dataset(platform="Twitter", dataset="tweets_br", class_label="True")
    print("--------------------------")
    print()    


if __name__ == "__main__":
    setup()
