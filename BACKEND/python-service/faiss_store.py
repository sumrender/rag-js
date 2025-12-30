"""
FAISS Store for vector storage and similarity search
Local vector database for embeddings storage and similarity search
"""
import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FAISSStore:
    """FAISS-based vector store with metadata and document storage"""
    
    def __init__(self, store_name: str, dimension: int, data_dir: str = "./faiss-data"):
        """
        Initialize FAISS store
        
        Args:
            store_name: Name of the store (e.g., "text_collection", "images_collection")
            dimension: Dimension of embeddings
            data_dir: Directory to store index and metadata files
        """
        self.store_name = store_name
        self.dimension = dimension
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # FAISS index (using IndexFlatL2 for L2 distance)
        self.index: Optional[faiss.Index] = None
        
        # Metadata storage: id -> metadata dict
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        # Document storage: id -> document string
        self.documents: Dict[str, str] = {}
        
        # ID to index mapping: id -> index position in FAISS
        self.id_to_index: Dict[str, int] = {}
        
        # Index to ID mapping: index position -> id
        self.index_to_id: Dict[int, str] = {}
        
        # File paths
        self.index_path = self.data_dir / f"{store_name}.index"
        self.metadata_path = self.data_dir / f"{store_name}.metadata.json"
        self.documents_path = self.data_dir / f"{store_name}.documents.json"
        self.mapping_path = self.data_dir / f"{store_name}.mapping.json"
        
        # Load existing data if available
        self.load()
    
    def _ensure_index(self):
        """Create FAISS index if it doesn't exist"""
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            logger.info(f"Created new FAISS index for {self.store_name} with dimension {self.dimension}")
    
    def upsert(self, ids: List[str], embeddings: List[List[float]], 
               documents: List[str], metadatas: List[Dict[str, Any]]):
        """
        Upsert (insert or update) embeddings with metadata and documents
        
        Args:
            ids: List of unique IDs
            embeddings: List of embedding vectors
            documents: List of document strings
            metadatas: List of metadata dictionaries
        """
        if not ids or not embeddings:
            return
        
        if len(ids) != len(embeddings) or len(ids) != len(documents) or len(ids) != len(metadatas):
            raise ValueError("ids, embeddings, documents, and metadatas must have the same length")
        
        self._ensure_index()
        
        # Convert embeddings to numpy array
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # Handle updates and new inserts
        embeddings_to_add = []
        ids_to_add = []
        metadatas_to_add = []
        documents_to_add = []
        
        for i, id_str in enumerate(ids):
            if id_str in self.id_to_index:
                # Update existing: update metadata and documents, add new embedding
                # Note: Old embedding remains in index but mapping points to new one
                # This is acceptable for local dev; for production, consider rebuilding index
                logger.debug(f"Updating existing entry: {id_str}")
                # Update metadata and documents immediately
                self.metadata[id_str] = metadatas[i]
                self.documents[id_str] = documents[i]
            
            # Add to new entries (including updates - new embedding will be added)
            embeddings_to_add.append(embeddings_array[i])
            ids_to_add.append(id_str)
            metadatas_to_add.append(metadatas[i])
            documents_to_add.append(documents[i])
        
        # Add embeddings to FAISS index
        if embeddings_to_add:
            embeddings_np = np.array(embeddings_to_add, dtype=np.float32)
            self.index.add(embeddings_np)
            
            # Update mappings (point to new indices)
            start_idx = len(self.id_to_index)
            for i, id_str in enumerate(ids_to_add):
                idx = start_idx + i
                # Update mapping to point to new index position
                self.id_to_index[id_str] = idx
                self.index_to_id[idx] = id_str
                # Update metadata/documents (already done for existing IDs, but do it again for consistency)
                self.metadata[id_str] = metadatas_to_add[i]
                self.documents[id_str] = documents_to_add[i]
        
        # Save to disk
        self.save()
        logger.info(f"Upserted {len(ids)} entries to {self.store_name}")
    
    def query(self, query_embeddings: List[List[float]], n_results: int, 
              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Query the store for similar vectors
        
        Args:
            query_embeddings: List of query embedding vectors
            n_results: Number of results to return
            where: Optional metadata filter (e.g., {"fileId": {"$eq": "some_id"}})
        
        Returns:
            Dictionary with keys: ids, documents, metadatas, distances
        """
        if self.index is None or self.index.ntotal == 0:
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]]
            }
        
        # Convert query embeddings to numpy
        query_array = np.array(query_embeddings, dtype=np.float32)
        
        # Query with larger k to account for filtering
        k = n_results * 3 if where else n_results
        k = min(k, self.index.ntotal)  # Don't query more than available
        
        # Search in FAISS
        distances, indices = self.index.search(query_array, k)
        
        # Process results for first query (assuming single query)
        result_indices = indices[0]
        result_distances = distances[0]
        
        # Get IDs, documents, and metadatas
        result_ids = []
        result_docs = []
        result_metas = []
        result_dists = []
        
        logger.debug(f"Query returned {len(result_indices)} results from FAISS, index has {self.index.ntotal} vectors, mappings have {len(self.index_to_id)} entries")
        
        for idx, dist in zip(result_indices, result_distances):
            if idx == -1:  # FAISS returns -1 for invalid indices
                continue
            
            if idx not in self.index_to_id:
                logger.debug(f"Index {idx} not found in index_to_id mapping (type: {type(idx)}, available keys: {list(self.index_to_id.keys())[:5]}...)")
                continue
            
            id_str = self.index_to_id[idx]
            
            # Apply metadata filter if provided
            if where:
                metadata = self.metadata.get(id_str, {})
                if not self._matches_filter(metadata, where):
                    continue
            
            result_ids.append(id_str)
            result_docs.append(self.documents.get(id_str, ""))
            result_metas.append(self.metadata.get(id_str, {}))
            result_dists.append(float(dist))
            
            # Stop if we have enough results
            if len(result_ids) >= n_results:
                break
        
        return {
            "ids": [result_ids],
            "documents": [result_docs],
            "metadatas": [result_metas],
            "distances": [result_dists]
        }
    
    def _matches_filter(self, metadata: Dict[str, Any], where: Dict[str, Any]) -> bool:
        """
        Check if metadata matches the where filter
        
        Args:
            metadata: Metadata dictionary to check
            where: Filter dictionary (e.g., {"fileId": {"$eq": "some_id"}})
        
        Returns:
            True if metadata matches filter
        """
        for key, condition in where.items():
            if key not in metadata:
                return False
            
            # Handle $eq operator
            if "$eq" in condition:
                if metadata[key] != condition["$eq"]:
                    return False
            # Add more operators as needed (e.g., $ne, $in, etc.)
        
        return True
    
    def save(self):
        """Save index, metadata, and documents to disk"""
        try:
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_path))
            
            # Save metadata
            with open(self.metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
            
            # Save documents
            with open(self.documents_path, 'w') as f:
                json.dump(self.documents, f, indent=2)
            
            # Save ID mappings (convert integer keys to strings for JSON)
            mapping_data = {
                "id_to_index": self.id_to_index,
                "index_to_id": {str(k): v for k, v in self.index_to_id.items()}
            }
            with open(self.mapping_path, 'w') as f:
                json.dump(mapping_data, f, indent=2)
            
            logger.debug(f"Saved {self.store_name} to disk")
        except Exception as e:
            logger.error(f"Failed to save {self.store_name}: {e}")
    
    def load(self):
        """Load index, metadata, and documents from disk"""
        try:
            # Load FAISS index
            if self.index_path.exists():
                self.index = faiss.read_index(str(self.index_path))
                logger.info(f"Loaded FAISS index from {self.index_path} ({self.index.ntotal} vectors)")
            else:
                self.index = None
            
            # Load metadata
            if self.metadata_path.exists():
                with open(self.metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            # Load documents
            if self.documents_path.exists():
                with open(self.documents_path, 'r') as f:
                    self.documents = json.load(f)
            
            # Load ID mappings
            if self.mapping_path.exists():
                with open(self.mapping_path, 'r') as f:
                    mapping_data = json.load(f)
                    self.id_to_index = mapping_data.get("id_to_index", {})
                    # Convert index_to_id keys from strings to integers (JSON loads them as strings)
                    raw_index_to_id = mapping_data.get("index_to_id", {})
                    self.index_to_id = {}
                    for key, value in raw_index_to_id.items():
                        try:
                            int_key = int(key)
                            self.index_to_id[int_key] = value
                        except (ValueError, TypeError):
                            # If key can't be converted, skip it
                            logger.warning(f"Skipping invalid index key in mapping: {key}")
            
            # Rebuild index_to_id from id_to_index if needed
            if not self.index_to_id and self.id_to_index:
                self.index_to_id = {v: k for k, v in self.id_to_index.items()}
            
            logger.info(f"Loaded {self.store_name}: {len(self.metadata)} entries")
        except Exception as e:
            logger.warning(f"Failed to load {self.store_name} from disk (will create new): {e}")
            self.index = None
            self.metadata = {}
            self.documents = {}
            self.id_to_index = {}
            self.index_to_id = {}

