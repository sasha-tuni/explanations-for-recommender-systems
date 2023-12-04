"""
DATA.ML.360-2023-2024-1 - Recommender Systems
Assignment 4 - Explanations for Why-not Questions in Recommender Systems
Sachini Hewage (152258085) & Robin Ivan Villa Soto (151814365)
December 4, 2023
"""

from group_recommendations import *
import numpy as np


def check_if_movie_available(movie_name, moviedata):
    """
    This movie checks if the movie is found in the dataset at all.
    If it is, it asks the user to select the correct movie from the list,
    or to confirm that it doesn't exist.
    :param movie_name: str, the name of the movie
    :param moviedata: df, the list of movies available
    :return: True and the movie ID if found, False if not found
    """
    movie = movie_name.lower()
    movie_tokens = movie.split(" ")

    filtered = moviedata.copy()

    for token in movie_tokens:
        filtered = filtered.loc[filtered['title'].str.contains(token)]

    del filtered['genres']

    if filtered.empty:
        return False, 0

    print("The following movies match your search:")
    print(filtered)
    print()
    print("Type the movie ID of the movie you are asking about")
    movie_id = input(
        "or 'q' if it does not appear in the list: ")
    print()

    if movie_id.isnumeric():
        movie_id = int(movie_id)
        return True, movie_id

    else:
        return False, 0


def check_if_nobody_rated(movie, ratings):
    """
    This function checks if any user in our dataset has rated the movie
    :param movie: str, the movie index of the movie we want
    :param ratings: df, containing all the ratings
    :return: True if nobody has rated it, False otherwise
    """
    try:
        return ratings[movie].isnull().all()
    except KeyError:
        return True


def check_if_in_group_ratings(movie, group, ratings):
    in_ratings = False
    users_that_have_rated = []

    for user in group:
        rating = ratings.at[user, movie]

        if not np.isnan(rating):
            in_ratings = True
            users_that_have_rated.append(rating)

    return in_ratings, users_that_have_rated


def check_if_similar_ratings(group, ratings, avg_df, movie):
    """
    This function checks if the similar users to each member of the group have
    provided ratings for a particular movie.
    If, for at least one user, none of the similar users provided a rating,
    it returns false (as then we cannot recommend this movie).
    It also returns a list with the members of the group for which none of
    their similar users provided ratings for the movie.
    Else, it returns true
    :param group: list, the group members
    :param ratings: df, containing all the ratings
    :param avg_df: df, containing the average ratings of all users
    :param movie: int, the movie index of the movie in question
    :return: Bool and list, see above
    """

    who_no_ratings = []

    for user in group:
        top_similar = get_top(user, 20, ratings, avg_df)
        similar_list = list(top_similar.index.values)

        in_similar, who = check_if_in_group_ratings(movie, similar_list,
                                                    ratings)

        if not in_similar:
            who_no_ratings.append(user)

    if len(who_no_ratings) > 0:
        return False, who_no_ratings

    else:
        return True, 0


def check_if_tie(movie, recs, n):
    """
    This function checks if the score of a movie in the recommended list
    matches the score of the nth item
    :param movie: int, the movie id
    :param recs: df, containing all the recommendable movies and their scores
    :param n: int, the position we are interested in comparing
    :return: Bool, True if there is a match, False otherwise
    """
    movie_rating = recs.loc[movie].values[0]
    nth_rating = recs.iloc[n - 1].values[0]

    if movie_rating == nth_rating:
        return True
    else:
        return False


def get_pos_rating(movie, recs):
    """
    This function checks where the movie of interest was ranked in the group
    recommendations. It also returns the predicted rating
    :param movie: int, the movie id
    :param recs: df, containing all the recommendable movies and their scores
    :return: int, the position of the movie in the list, also the rating
    """
    position = recs.index.get_loc(movie) + 1
    rating = recs.loc[movie].values[0]

    return position, rating


def explanation_for_absenteeism(movie, movies, ratings, group, avg_df,
                                top_picks, number_of_recommendations):
    found, movie_id = check_if_movie_available(movie, movies)

    # Checking if the movie exists in our database
    if found:
        not_rated_anyone = check_if_nobody_rated(movie_id, ratings)

        # Checking if any user in our dataset has rated it
        if not_rated_anyone:
            print("Nobody has rated this movie yet, so we cannot recommend it")

        # Checking if any member of the group has rated it
        else:
            in_ratings, who = check_if_in_group_ratings(movie_id, group,
                                                        ratings)
            if in_ratings:
                print(
                    f"The following users have already watched it, so it was "
                    f"not possible to recommend it: {who}")

            # Checking if any of the similar users have rated it
            else:
                in_similar, no_simil = check_if_similar_ratings(group, ratings,
                                                                avg_df,
                                                                movie_id)
                if not in_similar:
                    for user in no_simil:
                        print(f"None of the users similar to user {user} have "
                              f"rated the movie, ")
                        print(f"   so we cannot recommend it to "
                              f"that user, or to the group.")

                # Checking if the score is tied with the last recommended item
                else:
                    tie = check_if_tie(movie_id, top_picks,
                                       number_of_recommendations)
                    if tie:
                        print("This movie has the same score as the last item "
                              "in the list we provided. It was not "
                              "recommended to you because of how ties were "
                              "handled when processing the list.")
                    else:
                        position, rating = get_pos_rating(movie_id, top_picks)
                        print(f"With a predicted rating of {rating:.2f}, this "
                              f"movie is ")
                        print(f"ranked at position {position} of the list of "
                              f"possible ")
                        print("recommendations, so it was not suggested in "
                              "your ")
                        print(f"top {number_of_recommendations} list.")

    else:
        print("This movie was not suggested because it is not in our database")


def explanation_for_positional_absenteesm(movie_name, questioned_loc,
                                          group_no_na, means_formatted):
    # Normalise the title in the recommendations for the group
    means_formatted["normalised_title"] = means_formatted["title"].apply(
        lambda x: x.lower())

    filtered = pd.DataFrame

    # Find Das Boot
    movie = movie_name.lower()
    movie_tokens = movie.split(" ")
    for token in movie_tokens:
        filtered = means_formatted.loc[
            means_formatted['normalised_title'].str.contains(token)]

    # Find this movie's ID
    this_movie_id = filtered.index

    # Get this movie's predicted rating
    prediction_for_this_movie = filtered["Predicted Value"].values[0]

    # Get questioned location's movies predicted rating
    prediction_for_questioned = means_formatted.iloc[questioned_loc - 1][
        "Predicted Value"]

    # Find the position of this movie in the recommendations for the group
    rank_of_this = means_formatted.index.get_loc(filtered.index.values[0])

    # If the scores are equal, it is tie break issue
    if prediction_for_this_movie == prediction_for_questioned:
        print("The movie is in this position because of the way we")
        print("handled ties between movies with the same score.")

    # If this movie is ranked lower than the questioned location
    elif rank_of_this > questioned_loc - 1:
        group_results = group_no_na.loc[this_movie_id]
        # print(group_results)
        print("This is why the movie was presented at a lower position:")
        print(f" - The movie at position {questioned_loc} has a predicted "
              f"rating of {prediction_for_questioned:.2f}")
        print(f" - We predicted that the following users in your group")
        print(f"   would give '{movie}' a rating lower than the")
        print(f"   movie at position {questioned_loc}:")
        columns_above_threshold = (
                group_results < prediction_for_questioned).any().to_list()
        results = group_results.loc[:, columns_above_threshold]
        problem_users = results.columns.tolist()
        scores = results.iloc[0, :].values.flatten().tolist()

        for i in range(len(problem_users)):
            print(f"      - User {problem_users[i]}'s score for this movie: "
                  f"{scores[i]:.2f}")
        print(f"These predicted scores made '{movie}' rank lower than you "
              f"expected.")

    # If this movie is higher lower than the questioned location
    elif rank_of_this < questioned_loc - 1:
        group_results = group_no_na.loc[this_movie_id]
        # print(group_results)
        print("This is why the movie was presented at a higher position "
              "than expected:")
        print(f" - The movie at position {questioned_loc} has a predicted "
              f"rating of {prediction_for_questioned:.2f}")
        print(f" - We predicted that the following users in your group")
        print(f"   would give '{movie}' a rating higher than the ")
        print(f"   movie at position {questioned_loc}:")
        columns_above_threshold = (
                group_results > prediction_for_questioned).any().to_list()
        results = group_results.loc[:, columns_above_threshold]
        problem_users = results.columns.tolist()
        scores = results.iloc[0, :].values.flatten().tolist()

        for i in range(len(problem_users)):
            print(f"      - User {problem_users[i]}'s score for this movie: "
                  f"{scores[i]:.2f}")
        print(f"These predicted scores made '{movie}' rank higher than you "
              f"expected.")


def group_granularity_explainer(results, movies_of_this_genre,
                                group_prediction_means_all,
                                group_no_na, last, user_list, genre):
    # Check whether the initial ratings  dataset contains this genre in the
    # ratings data at all
    if results.any():
        ratings_temp = pd.read_csv("ratings.csv")
        ratings_data = pd.pivot_table(index="movieId",
                                      columns="userId",
                                      values="rating",
                                      data=ratings_temp)
        # Find the movie IDs in the ratings dataset
        ratings_indexes = ratings_data.index.to_list()

        # find the movies of this genre in the ratings data by getting the
        # intersection
        common_movies = list(
            set(ratings_indexes) & set(movies_of_this_genre))
        ratings_of_this_genre = ratings_data.loc[common_movies]

        # When you drop the all null rows from the ratings data of this genre,
        # if it is empty, we do not have any ratings for this genre.
        if ratings_of_this_genre.dropna(how='all').empty:
            print(f" We do not have any ratings for {genre} movies.")

        # If we do have any rated movies of this genre
        else:
            # Check if there is at least one movie no one has rated that would
            # make it a possibility to appear on the group recommendations
            if ratings_of_this_genre[user_list].isna().all(
                    axis=1).any():

                # Find all movies where all users in the group has predicted
                # ratings
                movies_with_group_ratings = group_no_na.index.to_list()

                # find all movies of this genre where all users in the group
                # has predicted ratings by taking the intersection
                group_ratings_of_this_genre = list(
                    set(movies_with_group_ratings) & set(
                        movies_of_this_genre))

                # If the list of movies from this genre in the base dataset
                # for group recommendations is empty, it means we do not have
                # any movies of this genre for this group of people to be
                # suggested.
                if len(group_ratings_of_this_genre) == 0:
                    print(f"There is not enough data to provide a rating for"
                          f" any {genre} movie for all users in the group.")

                # If we do have a movie of this genre with a possibility of
                # appearing in the recommendations for the group
                else:

                    # Get the last of TopN dataset's rating
                    ratings_for_last_suggestion = \
                        group_prediction_means_all.iloc[last - 1][
                            "Predicted Value"]

                    # find the first movie of this genre in all
                    # recommendations for this group
                    # (may or may not be in topN)
                    first_movie = group_prediction_means_all[
                        group_prediction_means_all.index.isin(
                            group_ratings_of_this_genre)].index[0]

                    # Find the integer index and the predicted rating of the
                    # first movie of this genre in the recommendations
                    first_index_iloc = group_prediction_means_all.index.get_loc(
                        first_movie)
                    ratings_for_first_of_this = \
                        group_prediction_means_all.iloc[first_index_iloc][
                            "Predicted Value"]

                    # If the first movie of this genre has the same rating as
                    # the last movie on the TOpN list and the user questions
                    # its absence, then it is a tie break issue.
                    if (
                            ratings_for_first_of_this == ratings_for_last_suggestion):
                        print(f"This is a tie-breaking issue! A {genre} movie")
                        print("had the same rating as the last item in the")
                        print("recommendations list")

                    # If not, present the highest ranked movie of this genre
                    # to the user with its current rank and predicted rating.
                    else:
                        print(f"The highest rated {genre} movie is "
                              f"found at position {first_index_iloc + 1} which"
                              f" has the following rating: "
                              f"{group_prediction_means_all.iloc[first_index_iloc].values[0]:.2f}")
                        print(f"so it did not make the top {last} "
                              f"recommendations")

            # If everybody has already rated (i.e. watched) all possible
            # movies of this genre
            else:
                print(
                    f"There are no {genre} movies that no one "
                    f"in the group has not watched!")

    # No results found in the dataset for this entered genre
    else:
        print("We do not have any movies of this genre to present!")


def main():
    # Editable Variables
    group = [598, 210, 400]
    number_of_recommendations = 10
    movies = pd.read_csv("movies.csv", index_col='movieId')

    # Setup
    movies['title'] = movies['title'].apply(lambda x: x.lower())
    ratings_temp = pd.read_csv("ratings.csv")
    ratings = pd.pivot_table(index="userId", columns="movieId",
                             values="rating", data=ratings_temp)
    avg_df = pd.DataFrame()
    avg_df['mean'] = ratings.mean(numeric_only=True, axis=1)
    group_no_na = get_group_no_na(group)

    # Recommending movies
    top_picks = group_preds(group, 10000)
    display_movies = top_picks.iloc[0:number_of_recommendations]
    top_movies = format_output(display_movies, movies)

    # Display the recommended movies
    print(top_movies)
    print()

    # Asking if they require why not explanations
    explanation_needed = input("Would you like to question the results? "
                               "[Y/N] ")
    print()

    # Explanations for why not questions
    if explanation_needed.lower() == "y":
        print("These are the kinds of questions we can answer: ")
        print()
        print("    1 Why is a movie missing?")
        print("    2 Why is this movie positioned in this ranking?")
        print("    3 Why are there no movies of a particular genre?")
        print()
        type_of_issue = input("Enter the number corresponding to the issue: ")
        print()

        # Absolute Absenteeism
        if type_of_issue == "1":
            movie = input("Which movie is missing? ")
            print()
            explanation_for_absenteeism(movie, movies, ratings, group, avg_df,
                                        top_picks, number_of_recommendations)

        elif type_of_issue == "2":
            movie = input("Which movie is in the wrong position? ")
            print()
            position = int(input("Which position should it go in? "))
            print()
            explanation_for_positional_absenteesm(movie, position, group_no_na,
                                                  top_movies)
        elif type_of_issue == "3":
            # Read genre_info dataset and find all movies of a selected genre
            genre_data = pd.read_csv("movies.csv")
            genre = input("Enter a genre : ")
            results = genre_data['genres'].str.contains(genre, case=False)

            # Find movie IDs of this genre using the gener_info dataset
            movies_of_this_genre = genre_data[
                genre_data['genres'].str.contains(genre, case=False)][
                "movieId"].to_list()

            # Group predictions using means method for all possibilities
            group_prediction_means_all = mean_rating(group_no_na, 10000,
                                                     'means')

            group_granularity_explainer(results, movies_of_this_genre,
                                        group_prediction_means_all,
                                        group_no_na, number_of_recommendations,
                                        group, genre)

        else:
            print("Invalid input. Have a nice day.")
    else:
        print("Enjoy your movies!")


if __name__ == "__main__":
    main()
