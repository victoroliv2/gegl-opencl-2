dnl GEGL_DETECT_CFLAGS(RESULT, FLAGSET)
dnl Detect if the compiler supports a set of flags

AC_DEFUN([GEGL_DETECT_CFLAGS],
[
  $1=
  for flag in $2; do
    if test -z "[$]$1"; then
      $1_save_CFLAGS="$CFLAGS"
      CFLAGS="$CFLAGS $flag"
      AC_MSG_CHECKING([whether [$]CC understands [$]flag])
      AC_TRY_COMPILE([], [], [$1_works=yes], [$1_works=no])
      AC_MSG_RESULT([$]$1_works)
      CFLAGS="[$]$1_save_CFLAGS"
      if test "x[$]$1_works" = "xyes"; then
        $1="$flag"
      fi
    fi
  done
])


dnl The following lines were copied from gtk-doc.m4

dnl Usage:
dnl   GTK_DOC_CHECK([minimum-gtk-doc-version])
AC_DEFUN([GTK_DOC_CHECK],
[
  AC_BEFORE([AC_PROG_LIBTOOL],[$0])dnl setup libtool first
  AC_BEFORE([AM_PROG_LIBTOOL],[$0])dnl setup libtool first
  dnl for overriding the documentation installation directory
  AC_ARG_WITH(html-dir,
    AC_HELP_STRING([--with-html-dir=PATH], [path to installed docs]),,
    [with_html_dir='${datadir}/gtk-doc/html'])
  HTML_DIR="$with_html_dir"
  AC_SUBST(HTML_DIR)

  dnl enable/disable documentation building
  AC_ARG_ENABLE(gtk-doc,
    AC_HELP_STRING([--enable-gtk-doc],
                   [use gtk-doc to build documentation (default=no)]),,
    enable_gtk_doc=no)

  have_gtk_doc=no
  if test x$enable_gtk_doc = xyes; then
    if test -z "$PKG_CONFIG"; then
      AC_PATH_PROG(PKG_CONFIG, pkg-config, no)
    fi
    if test "$PKG_CONFIG" != "no" && $PKG_CONFIG --exists gtk-doc; then
      have_gtk_doc=yes
    fi

  dnl do we want to do a version check?
ifelse([$1],[],,
    [gtk_doc_min_version=$1
    if test "$have_gtk_doc" = yes; then
      AC_MSG_CHECKING([gtk-doc version >= $gtk_doc_min_version])
      if $PKG_CONFIG --atleast-version $gtk_doc_min_version gtk-doc; then
        AC_MSG_RESULT(yes)
      else
        AC_MSG_RESULT(no)
        have_gtk_doc=no
      fi
    fi
])
    if test "$have_gtk_doc" != yes; then
      enable_gtk_doc=no
    fi
  fi

  AM_CONDITIONAL(ENABLE_GTK_DOC, test x$enable_gtk_doc = xyes)
  AM_CONDITIONAL(GTK_DOC_USE_LIBTOOL, test -n "$LIBTOOL")
])
