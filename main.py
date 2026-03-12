import kagglehub 
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("wardabilal/spotify-global-music-dataset-20092025")
df = pd.read_csv(path + "/track_data_final.csv" , sep=",")

print(df.head(10))
print(df.info())

print("\n")
colunaPopulariedadeMusica = df["track_popularity"]
mediaPopulariedade = colunaPopulariedadeMusica.mean()
medianaPopulariedade = colunaPopulariedadeMusica.median()
modaPopulariedade = colunaPopulariedadeMusica.mode().tolist()
varAmostral_PopulariedadeMusica = colunaPopulariedadeMusica.var(ddof=1)
dpAmostral_PopulariedadeMusica = colunaPopulariedadeMusica.std(ddof=1)
print(f"Média da popularidade: {mediaPopulariedade:.2f}")
print(f"Mediana da populariade: {medianaPopulariedade}")
print(f"Moda da popularidade: {modaPopulariedade}")
print(f"Variância da popularidade: {varAmostral_PopulariedadeMusica:.2f}")
print(f"Desvio Padrão da popularidade: {dpAmostral_PopulariedadeMusica:.2f}")

plt.hist(colunaPopulariedadeMusica, bins="auto")
plt.title('Histograma da Popularidade das Músicas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
plt.show()



print("\n")
colunaPopulariedadeArtista = df["artist_popularity"]
mediaPopulariedadeArtista = colunaPopulariedadeArtista.mean()
medianaPopulariedadeArtista = colunaPopulariedadeArtista.median()
modaPopulariedadeArtista = colunaPopulariedadeArtista.mode().tolist()
varAmostral_PopulariedadeArtista = colunaPopulariedadeArtista.var(ddof=1)
dpAmostral_PopulariedadeArtista = colunaPopulariedadeArtista.std(ddof=1)
print(f"Média da popularidade do artista: {mediaPopulariedadeArtista:.2f}")
print(f"Mediana da popularidade do artista: {medianaPopulariedadeArtista}")
print(f"Moda da popularidade do artista: {modaPopulariedadeArtista}")
print(f"Variância da popularidade do artista: {varAmostral_PopulariedadeArtista:.2f}")
print(f"Desvio Padrão da popularidade do artista: {dpAmostral_PopulariedadeArtista:.2f}")

plt.hist(colunaPopulariedadeArtista, bins="auto", edgecolor='black')
plt.title('Histograma da Popularidade dos Artistas')
plt.xlabel('Popularidade')
plt.ylabel('Frequência')
plt.show()