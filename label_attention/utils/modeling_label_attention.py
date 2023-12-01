"""
WARNING: Not implemented. 
Should eventually become an abstract class for label embedding.
"""

class BertLabelEmbedding(nn.Module):
    def __init__(self, config, pretrained_model=None, pretrained_tokenizer=None):
        super(BertLabelEmbedding, self).__init__()
        logger.info("Initializing label embedding")
        self.num_labels = config.num_labels
        self.config = config

        # Set this flag if pretrained weights will be loaded
        self.pretrained_weights = False

        if not self.pretrained_weights:
            self.init_weights(pretrained_model, pretrained_tokenizer)
        else:
            self.label_attention_matrix = nn.Parameter(torch.zeros((self.num_labels, 
                                                                    self.config.hidden_dim)))  # n_labels x hidden_dim
    
    def init_weights(self, weights = None, model=None, tokenizer=None):
        """
        Initialize label embedding matrix with sentence embeddings of labels.
        Hidden size of config/model should match target model. 
        """
        
        # Tokenize and encode label names
        self.label_names = self.config.label2idx.keys()

        if model is not None and tokenizer is not None:
            tokenized_labels = tokenizer(self.label_names)
        
            # Get hidden representation of label encodings
            model.eval()
            with torch.no_grad():
                init_embeddings = model(tokenized_labels)[1] # CLS TOKEN OUT
        
            self.label_attention_matrix = nn.Parameter(init_embeddings)    # n_labels x hidden_dim

    def forward(self, inputs):
        """
        Calculate cosine distance between each sample and each label
        
        args
            inputs: Tensor of shape (N x hidden_dim)
        returns
            tensor of shape N x num_classes
        """

        # Compute cosine similarity matrix
        eps = 1e-8
        X_n = X.norm(dim=1)[:, None]
        W_n = self.label_attention_matrix.norm(dim=1)[:, None]
        X_norm = X / torch.where(W_n < eps, W_n)
        W_norm = self.label_attention_matrix / torch.where(W_n < eps, eps, W_n)
        cos_sim = torch.mm(X_norm, self.label_attention_matrix.T)

        return cos_sim
