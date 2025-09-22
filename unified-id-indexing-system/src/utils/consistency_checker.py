from typing import List, Dict
from db.database_manager import db_manager
from core.unified_id import UnifiedID

class ConsistencyChecker:
    """Checks the consistency of data across different databases based on unified IDs."""

    def __init__(self):
        self.db_manager = db_manager

    def check_unified_id_existence(self, unified_id: str) -> bool:
        """Check if the unified ID exists across all databases."""
        mongo_exists = self.db_manager.mongo.check_unified_id(unified_id)
        neo4j_exists = self.db_manager.neo4j.check_unified_id(unified_id)
        redis_exists = self.db_manager.redis.check_unified_id(unified_id)
        milvus_exists = self.db_manager.milvus.check_unified_id(unified_id)

        return mongo_exists and neo4j_exists and redis_exists and milvus_exists

    def report_inconsistencies(self, unified_ids: List[str]) -> Dict[str, List[str]]:
        """Report inconsistencies for a list of unified IDs."""
        inconsistencies = {
            "missing_in_mongo": [],
            "missing_in_neo4j": [],
            "missing_in_redis": [],
            "missing_in_milvus": []
        }

        for unified_id in unified_ids:
            if not self.db_manager.mongo.check_unified_id(unified_id):
                inconsistencies["missing_in_mongo"].append(unified_id)
            if not self.db_manager.neo4j.check_unified_id(unified_id):
                inconsistencies["missing_in_neo4j"].append(unified_id)
            if not self.db_manager.redis.check_unified_id(unified_id):
                inconsistencies["missing_in_redis"].append(unified_id)
            if not self.db_manager.milvus.check_unified_id(unified_id):
                inconsistencies["missing_in_milvus"].append(unified_id)

        return inconsistencies

    def check_all_unified_ids(self) -> None:
        """Check all unified IDs in the system for consistency."""
        unified_ids = self.db_manager.mongo.get_all_unified_ids()
        inconsistencies = self.report_inconsistencies(unified_ids)

        if any(inconsistencies.values()):
            print("Inconsistencies found:")
            for db, ids in inconsistencies.items():
                if ids:
                    print(f"{db}: {len(ids)} missing IDs")
        else:
            print("All unified IDs are consistent across databases.")