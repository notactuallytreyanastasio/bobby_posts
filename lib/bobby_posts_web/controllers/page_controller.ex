defmodule BobbyPostsWeb.PageController do
  use BobbyPostsWeb, :controller

  def home(conn, _params) do
    render(conn, :home)
  end
end
