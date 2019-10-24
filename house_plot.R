my_data <- read_excel("truliaHouse(second cleaned).xlsx")
View(my_data)

ggplot(my_data, aes(x = my_data$Lat, y = my_data$Lon, color = price)) + 
  geom_jitter(alpha = 0.3) + 
  scale_color_gradientn(colours = rainbow(5), limits=c(100000,2000000)) + 
  xlim(32,32.5) + 
  ylim(-111.5, -110.5)

p <- plot_ly(my_data, x = ~Lat, y = ~Lon, z = ~price,
             marker = list(color = ~price, colorscale = c('#FFE1A1', '#683531'), 
                           cauto = F, cmin = 0, cmax = 2000000, showscale = TRUE)) %>%
  add_markers() %>%
  layout(scene = list(xaxis = list(title = 'Latitude', range=c(31.8,32.5)),
                      yaxis = list(title = 'Longitude', range=c(-111.5,-110.5)),
                      zaxis = list(title = 'Price', range=c(100000,2000000))),
         annotations = list(
           x = 1.13,
           y = 1.05,
           text = 'Price',
           xref = 'paper',
           yref = 'paper',
           showarrow = FALSE
         ))
p

#list(c(0, “rgb(255, 0, 0)”), list(1, “rgb(0, 255, 0)”)),
#c('#FFE1A1', '#683531')