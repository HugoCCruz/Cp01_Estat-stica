import kagglehub 
import pandas as pd
import matplotlib.pyplot as plt

path = kagglehub.dataset_download("wardabilal/spotify-global-music-dataset-20092025")
df = pd.read_csv(path + "/track_data_final.csv" , sep=",")

print(df.columns)

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



print("\n")
# par 1
x1 = df["artist_popularity"]
y1 = df["track_popularity"]
corr1 = x1.corr(y1)
print(f"Correlação (artist_popularity x track_popularity): {corr1:.3f}")

plt.scatter(x1, y1)
plt.xlabel("Popularidade dos artistas")
plt.ylabel("Popularidade da música")
plt.title("Scatterplot da Popularidade dos artistas x Popularidade da música")
plt.show()


# par 2
x2 = df["artist_followers"]
y2 = df["track_popularity"]
corr2 = x2.corr(y2)
print(f"Correlação (artist_followers x track_popularity): {corr2:.3f}")

plt.scatter(x2, y2)
plt.xlabel("Seguidores dos artistas")
plt.ylabel("Popularidade da música")
plt.title("Scatterplot dos Seguidores dos artistas x Popularidade da música")
plt.show()



print("\n")

col = df["track_popularity"].dropna()

Q1 = col.quantile(0.25)
Q3 = col.quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5 * IQR
high = Q3 + 1.5 * IQR
print(f"Limite inferior: {low}")
print(f"Limite superior: {high}")

outliers = (col < low) | (col > high)
print("Quantidade de outliers:", outliers.sum())

plt.boxplot(col)
plt.title("Boxplot da Popularidade da Música (antes)")
plt.ylabel("Popularidade da música")
plt.show()
col_sem_outliers = col[~outliers]
plt.boxplot(col_sem_outliers)
plt.title("Boxplot da Popularidade da Música (sem outliers)")
plt.ylabel("Popularidade da música")
plt.show()
