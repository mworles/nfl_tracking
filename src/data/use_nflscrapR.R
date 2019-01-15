library(nflscrapR)
s2017 <- season_play_by_play(2017)
save(s2017,file="C:/Users/mworley/nfl_tracking/data/raw/scrapR2017.Rda")
write.csv(s2017, file = "C:/Users/mworley/nfl_tracking/data/raw/scrapR2017.csv", row.names=FALSE)
