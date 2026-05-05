import tensorflow as tf
import numpy as np 

class VITEmbedding(tf.keras.layers.Layer):

    def __init__(self, patching_size, embed_dim, linear_projection=tf.keras.layers.Dense):
        super().__init__()

        self.patching_size = patching_size
        self.linear_projection = tf.keras.layers.Dense(embed_dim)
        self.flatten = tf.keras.layers.Flatten()
        self.pos_embedding = self.add_weight(
            shape=(1, patching_size, embed_dim),
            initializer="random_normal",
            trainable=True
        )

    def call(self, spectrogram):

        patches = tf.split(spectrogram,num_or_size_splits=self.patching_size,axis=1)

        processed_patches = []

        for patch in patches:
            patch = self.flatten(patch)
            patch = self.linear_projection(patch)
            processed_patches.append(patch)

        embeddings = tf.stack(processed_patches, axis=1)
        embeddings = embeddings + self.pos_embedding

        return  embeddings



class transformEncoder(tf.keras.layers.Layer):

    def __init__(self, num_heads, embed_dim, mlp_ratio=4,**kwargs):
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = embed_dim

        
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

       
        self.attn = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim // num_heads
        )

       
        hidden_dim = embed_dim * mlp_ratio
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_dim, activation=tf.nn.gelu),
            tf.keras.layers.Dense(embed_dim)
        ])

    def call(self, x):

        x_norm = self.norm1(x)

        attn_out = self.attn(
            query=x_norm,
            value=x_norm,
            key=x_norm
        )

        x = x + attn_out   


        x_norm = self.norm2(x)

        mlp_out = self.mlp(x_norm)

        x = x + mlp_out   

        return x
    


class VIT(tf.keras.Model):

    def __init__(self, patching_size, embed_dim, num_heads, num_classes):
        super().__init__()

        self.embedding = VITEmbedding(patching_size, embed_dim)
        self.encoder = transformEncoder(num_heads, embed_dim)

       
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

       
        self.head = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, x):

        x = self.embedding(x)      
        x = self.encoder(x)        
        x = self.pool(x)           
        x = self.head(x)           

        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_classes": self.num_classes,
        })
        return config





