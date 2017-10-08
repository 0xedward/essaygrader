Rails.application.routes.draw do
    resources :documents, only: [:index, :new, :create, :destroy]
    root "documents#index"
  # For details on the DSL available within this file, see http://guides.rubyonrails.org/routing.html
end
