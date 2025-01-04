class SMILESTokenizer:
    def __init__(self):
        # Special tokens
        self.pad_token = "[PAD]"
        self.start_token = "[START]"
        self.end_token = "[END]"
        
        # Will store our vocabulary
        self.token_to_id = {}
        self.id_to_token = {}
        
        self.max_length = None  # Will be set when building vocabulary
        
    def build_vocabulary(self, smiles_list):
        """Create vocabulary from list of SMILES"""
        # Start with special tokens
        tokens = {self.pad_token, self.start_token, self.end_token}
        
        # Add all unique characters from SMILES
        for smiles in smiles_list:
            tokens.update(set(smiles))
        
        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(sorted(tokens))}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        
        print(f"Vocabulary size: {len(self.token_to_id)}") 
        
        # Set max length based on data
        # Add 2 for [START] and [END] tokens
        self.max_length = max(len(s) for s in smiles_list) + 2
        print(f"Max sequence length: {self.max_length}")
    
    def tokenize(self, smiles):
        """Convert a SMILES string to a list of tokens"""
        tokens = [self.start_token]  # Start with start token
        tokens.extend(list(smiles))  # Add SMILES characters
        tokens.append(self.end_token)  # End with end token
        
        # Pad if necessary
        while len(tokens) < self.max_length:
            tokens.append(self.pad_token)
            
        # Truncate if too long
        tokens = tokens[:self.max_length]
        
        # Convert to IDs
        token_ids = [self.token_to_id[token] for token in tokens]
        
        return token_ids
        