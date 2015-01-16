/* testing the spro header */

#include <stdio.h>

#include <spro.h>

void
print_spf_stream_t(spfstream_t *s)
{
     printf("--------------------\n");
     printf("name:        %s\n", s->name);
     printf("f:           N/A\n");
     printf("iomode:      %d\n", s->iomode);
     printf("Fs:          %.3f\n", s->Fs);
     printf("idim:        %hu\n", s->idim);
     printf("iflag:       %Ld\n", s->iflag);
     printf("odim:        %hu\n", s->odim);
     printf("oflag:       %Ld\n", s->oflag);
     printf("cflag:       %Ld\n", s->cflag);
     printf("winlen:      %Lu\n", s->winlen);
     printf("escale:      %.3f\n", s->escale);
     printf("start:       %Lu\n", s->start);
     printf("ix:          %Lu\n", s->idx);
     printf("--------------------\n");
}

spfbuf_t *fill_input_stream(const char *fname)
{
     spfstream_t *f = NULL;
     spf_t *spf = NULL;
     spfbuf_t *os = NULL;
     unsigned long int nframes;

     /* open feature stream */
     if ((f = spf_input_stream_open(fname, 0, 10000000)) == NULL) {
          fprintf(stderr, "fill_input_stream error() -- cannot open feature "
                  "stream %s\n", fname);
          return NULL;
     }
     print_spf_stream_t(f);

     float framerate = spf_stream_rate(f);
     unsigned short vecdim = spf_stream_dim(f);
     spf_stream_close(f);

     /* allocate memory */
     if (((os) = spf_buf_alloc(vecdim, 1)) == NULL) {
          return NULL;
     }
     if ((f = spf_input_stream_open(fname, 0, 10000000)) == NULL) {
          fprintf(stderr, "fill_input_stream error() -- cannot open feature "
                  "stream %s\n", fname);
          spf_buf_free(os);
          return NULL;
     }
     printf("header:\n");
     for (int i = 0; i < f->header->nfields; ++i) {
          char *name = f->header->field[i].name;
          char *value = f->header->field[i].value;
          printf("%s = %s\n", name, value);
     }

     print_spf_stream_t(f);

     nframes = 0;
     spf = get_next_spf_frame(f);
     /* print_spf_stream_t(f); */
     while (spf != NULL) {
          if ((spf_buf_append((os), spf, vecdim, 1)) == NULL) {
               spf_buf_free(os);
               return NULL;
          }
          spf = get_next_spf_frame(f);
          nframes++;
     }
     spf_stream_close(f);

     printf("------------\n");
     printf("post-header:\n");
     printf("------------\n");
     printf("vecdim:    %hu\n", vecdim);
     printf("framerate: %.3f\n", framerate);
     printf("----\n");
     printf("data\n");
     printf("----\n");
     for (int vec_ix = 0; vec_ix < os->n; ++vec_ix) {
          int start = vec_ix * vecdim;
          int end = start + vecdim;
          for (int feat_ix = start; feat_ix < end; ++feat_ix) {
               printf("%.3f ", os->s[feat_ix]);
          }
          printf("\n");
     }
}

int main()
{
     fill_input_stream("test.bin");
     return 0;
}
