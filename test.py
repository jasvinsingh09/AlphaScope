from src.sector_ranker import rank_sectors

sector_scores = rank_sectors()

for sector in sector_scores:
    print(sector)