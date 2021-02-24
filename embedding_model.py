import os
from typing import Any, Dict
from kashgari.embeddings import TransformerEmbedding


class AlbertEmbedding(TransformerEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(AlbertEmbedding, self).to_dict()
        info_dic['config']['model_folder'] = self.model_folder
        return info_dic

    def __init__(self,
                 model_folder: str,
                 **kwargs: Any):
        """

        Args:
            model_folder: path of checkpoint folder.
            kwargs: additional params
        """
        self.model_folder = model_folder
        vocab_path = os.path.join(self.model_folder, 'vocab.txt')
        config_path = os.path.join(self.model_folder, 'albert_config_large.json')
        checkpoint_path = os.path.join(self.model_folder, 'albert_model.ckpt')
        kwargs['vocab_path'] = vocab_path
        kwargs['config_path'] = config_path
        kwargs['checkpoint_path'] = checkpoint_path
        kwargs['model_type'] = 'albert'
        super(AlbertEmbedding, self).__init__(**kwargs)


class RoBertaEmbedding(TransformerEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(RoBertaEmbedding, self).to_dict()
        info_dic['config']['model_folder'] = self.model_folder
        return info_dic

    def __init__(self,
                 model_folder: str,
                 **kwargs: Any):
        """

        Args:
            model_folder: path of checkpoint folder.
            kwargs: additional params
        """
        self.model_folder = model_folder
        vocab_path = os.path.join(self.model_folder, 'vocab.txt')
        config_path = os.path.join(self.model_folder, 'bert_config.json')
        checkpoint_path = os.path.join(self.model_folder, 'bert_model.ckpt')
        kwargs['vocab_path'] = vocab_path
        kwargs['config_path'] = config_path
        kwargs['checkpoint_path'] = checkpoint_path
        kwargs['model_type'] = 'roberta'
        super(RoBertaEmbedding, self).__init__(**kwargs)


class RoBertaLargeEmbedding(TransformerEmbedding):
    def to_dict(self) -> Dict[str, Any]:
        info_dic = super(RoBertaLargeEmbedding, self).to_dict()
        info_dic['config']['model_folder'] = self.model_folder
        return info_dic

    def __init__(self,
                 model_folder: str,
                 **kwargs: Any):
        """

        Args:
            model_folder: path of checkpoint folder.
            kwargs: additional params
        """
        self.model_folder = model_folder
        vocab_path = os.path.join(self.model_folder, 'vocab.txt')
        config_path = os.path.join(self.model_folder, 'bert_config_large.json')
        checkpoint_path = os.path.join(self.model_folder, 'roberta_zh_large_model.ckpt')
        kwargs['vocab_path'] = vocab_path
        kwargs['config_path'] = config_path
        kwargs['checkpoint_path'] = checkpoint_path
        kwargs['model_type'] = 'roberta'
        super(RoBertaLargeEmbedding, self).__init__(**kwargs)