<?

    //Preprocessor to be included in every file for checking how the user is accessing the file
    define('_c4k', true);

    //Current Root directory specified
    define('_root', realpath(dirname(__FILE__)));
    
    //Including the framework to C4K project
    require_once ( _root . "/lib/framework.php" );
?>
