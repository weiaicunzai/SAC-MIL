" set nocompatible 
" 试一下
set backspace=indent,eol,start
"highlight Normal ctermfg=red ctermbg=black
"highlight Normal ctermbg=Blue
set nu
"set background=dark
syntax on
set wm=10
filetype off

set rtp+=~/.vim/bundle/vundle/
call vundle#rc()

" let Vundle manage Vundle
" required!

Bundle 'gmarik/vundle'
Bundle 'Lokaltog/powerline', {'rtp': 'powerline/bindings/vim/'}
"github tool
Bundle 'tpope/vim-fugitive'   
"folder tree                              
Bundle 'scrooloose/nerdtree'    
"autocomplete                           
Bundle 'klen/python-mode'                        
Bundle 'Valloric/YouCompleteMe'
Bundle 'altercation/vim-colors-solarized'

" The bundles you install will be listed here

filetype plugin indent on



" The rest of your config follows here


let g:ycm_autoclose_preview_window_after_completion=1
map <leader>g  :YcmCompleter GoToDefinitionElseDeclaration<CR>

syntax enable
set background=dark
colorscheme solarized
let g:solarized_termcolors=256

 set laststatus=2   " 保证powerline 总是出现
" let g:pymode_rope = 0
map <F2> :NERDTreeToggle<CR> 
" press F2 in vim and it will take you to the current working dirctory  
"   Press ? to see NerdTree's list of commands.

augroup vimrc_autocmds
    autocmd!
    " highlight characters past column 120
    autocmd FileType python highlight Excess ctermbg=DarkGrey guibg=Black
    autocmd FileType python match Excess /\%120v.*/
    autocmd FileType python set nowrap
    augroup END

