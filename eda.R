library(dplyr)
library(ggplot2)
library(ggcorrplot)
library(forcats)

data <- read.csv("TurismoCWB(1).csv")

# Plots:
# Quantidade de categorias

# Tratar a coluna extra
data_clean <- data %>%
  select(-X)

# quantidade de valores ausentes
nan_total <- colSums(
  is.na(data_clean) |
    data_clean == "none" |
    data_clean == "Nan" |
    data_clean == "nan"
  )

nan_total_df <- data.frame(
  column = names(nan_total),
  missing_count = nan_total
)

plot_nan <- ggplot(data = nan_total_df, aes(x=column, y=missing_count, fill = column))+
  geom_bar(stat="identity")+
  scale_fill_brewer(palette = "Set1")+
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 60, face = "italic", hjust = 1, vjust = 1)
    ) +
  labs(x = "Coluna", y = "Quantidade de valores ausentes por coluna",
       title = "Valores ausentes por coluna")

ggsave(plot_nan, filename = "plot_nan.svg", device = "svg", height = 10, width = 10)


# matriz de correlação entre variáveis

numeric_data <- data_clean[, sapply(data_clean, is.numeric)]

corr <- round(cor(numeric_data), 1)

matriz_cor <- ggcorrplot(corr, hc.order = T, outline.color = "white")

ggsave("matriz_cor.svg", plot = matriz_cor, device = "svg", height = 10, width = 10)


geom_esp <- ggplot(data_clean, aes(x = longitude, y = latitude)) +
geom_point(aes(color = categoria, size = popularidade), alpha = 0.8) +
  scale_size_continuous(range = c(3, 10)) +
  labs(
    title = "Mapa Geoespacial dos Pontos Turísticos",
    x = "Longitude",
    y = "Latitude",
    color = "Categoria",
    size = "Popularidade"
  ) +
  theme_minimal() +
  theme(legend.position = "right")

ggsave("mapa_geoespacial.png", plot = geom_esp, width = 12, height = 8)


bar_cat <- ggplot(data_clean, aes(x = fct_infreq(categoria), fill = categoria)) +
  geom_bar(show.legend = FALSE) +
  labs(
    title = "Contagem de Pontos Turísticos por Categoria",
    x = "Categoria",
    y = "Contagem"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("categoria_barplot.png", plot = bar_cat, width = 10, height = 6)
