import unittest
from datetime import datetime

import pandas as pd

from mm_utils import build_broker_ranking, extract_postal_code, load_oslo_postal_lookup


class PostalMappingTests(unittest.TestCase):
    def test_extract_postal_code_prefers_last_match(self):
        address = "Parkveien 76 A (h 0201, A 3), 0254 Oslo"
        self.assertEqual(extract_postal_code(address), "0254")

    def test_oslo_lookup_contains_expected_entries(self):
        lookup = load_oslo_postal_lookup()
        self.assertIn("0152", lookup)
        self.assertEqual(lookup["0152"], "Sentrum")
        self.assertEqual(lookup["0782"], "Vestre Aker")
        self.assertEqual(lookup["1087"], "Nordstrand")


class BrokerRankingTests(unittest.TestCase):
    def test_build_broker_ranking_simple(self):
        data = pd.DataFrame(
            {
                "broker": ["Alice", "Alice", "Bob"],
                "chain": ["Chain 1", "Chain 1", "Chain 2"],
                "listing_id": [1, 2, 3],
                "price": [5_000_000, 3_000_000, 4_000_000],
                "city": ["Oslo", "Oslo", "Oslo"],
                "district": ["Frogner", "Frogner", "Sagene"],
                "property_type": ["Leilighet", "Leilighet", "Enebolig"],
                "broker_role": ["Megler", "Megler", "Megler"],
                "published_dt": pd.to_datetime(
                    ["2024-01-10", "2024-03-15", "2024-02-01"], utc=True
                ),
            }
        )

        ranking = build_broker_ranking(data)
        self.assertEqual(len(ranking), 2)

        alice_row = ranking[ranking["broker"] == "Alice"].iloc[0]
        self.assertEqual(int(alice_row["total_sales"]), 2)
        self.assertEqual(alice_row["total_value"], 8_000_000)
        self.assertEqual(alice_row["avg_price"], 4_000_000)
        self.assertEqual(alice_row["dominant_segment"], "Leilighet")
        self.assertEqual(alice_row["primary_location"], "Frogner")
        self.assertTrue(alice_row["high_volume"])
        self.assertEqual(alice_row["broker_key"], "Alice||Chain 1")

        bob_row = ranking[ranking["broker"] == "Bob"].iloc[0]
        self.assertEqual(int(bob_row["total_sales"]), 1)
        self.assertEqual(bob_row["total_value"], 4_000_000)
        self.assertFalse(bob_row["high_volume"])


if __name__ == "__main__":
    unittest.main()
